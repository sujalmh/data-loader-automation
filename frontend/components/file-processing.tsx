"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Play,
  RotateCcw,
  FileText,
  CheckCircle,
  Clock,
  AlertCircle,
  Eye,
} from "lucide-react";
import type { FileData, AnalysisData } from "@/app/page";

interface ApiResult {
  fileName: string;
  qualityMetrics: {
    parseAccuracy: number;
  };
  classification: "Structured" | "Semi-Structured" | "Unstructured";
  analysis: AnalysisData;
}

interface FileProcessingProps {
  files: FileData[];
  setFiles: (updater: (prevFiles: FileData[]) => FileData[]) => void;
  progress: number;
  setProgress: (progress: number) => void;
}

export default function FileProcessing({
  files,
  setFiles,
  progress,
  setProgress,
}: FileProcessingProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processedFileCount, setProcessedFileCount] = useState(0);
  const [selectedFileForDetails, setSelectedFileForDetails] =
    useState<FileData | null>(null);

  const selectedFiles = files.filter((f) => f.selected);
  const failedFiles = selectedFiles.filter((f) => f.processed && f.error);
  const allFilesProcessed =
    selectedFiles.length > 0 && selectedFiles.every((f) => f.processed);

  const processFiles = async (filesToProcess: FileData[]) => {
    if (filesToProcess.length === 0) return;

    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    filesToProcess.forEach((file) => {
      if (file.file) {
        formData.append("files", file.file);
        formData.append("file_ids", file.id);
      }
    });

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const response = await fetch(`${apiUrl}/process-files`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errBody = await response.text().catch(() => "");
        throw new Error(
          `Upload failed: ${response.status} ${response.statusText} ${errBody}`
        );
      }
      if (!response.body) {
        throw new Error("No response body available for streaming.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const dataBlockRegex = /(?:^|\n)data:\s*((?:.|\n)*?)(?=\n\n|$)/g;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let match;
        let lastIndex = 0;
        dataBlockRegex.lastIndex = 0;
        while ((match = dataBlockRegex.exec(buffer)) !== null) {
          const rawData = match[1].trim();
          lastIndex = dataBlockRegex.lastIndex;

          try {
            const payload = JSON.parse(rawData);
            console.debug("SSE payload received:", payload);

            const fileId = payload.fileId ?? payload.file_id ?? null;
            if (!fileId) {
              console.warn("Payload missing fileId, skipping:", payload);
              continue;
            }

            let analysis = payload.analysis ?? null;
            if (
              analysis &&
              typeof analysis === "object" &&
              "json" in analysis
            ) {
              analysis = analysis.json;
            }

            setFiles((prevFiles) => {
              const updated = prevFiles.map((f) => {
                if (f.id === fileId) {
                  const parseAccuracy =
                    (analysis &&
                      (analysis.quality_score ?? analysis.quality)) ??
                    undefined;
                  const qualityMetrics =
                    typeof parseAccuracy === "number"
                      ? { parseAccuracy }
                      : undefined;

                  return {
                    ...f,
                    processed: true,
                    qualityMetrics,
                    classification: payload.analysis.classification ?? "Unstructured",
                    analysis: analysis ?? payload.analysis ?? payload,
                    error: payload.error ?? undefined,
                  };
                }
                return f;
              });

              const processedSoFar = updated.filter(
                (f) => f.selected && f.processed
              ).length;
              setProcessedFileCount(processedSoFar);
              setProgress(
                (processedSoFar / Math.max(1, selectedFiles.length)) * 100
              );

              return updated;
            });
          } catch (err) {
            console.error("Failed to parse SSE JSON:", rawData, err);
          }
        }
        buffer = buffer.slice(lastIndex);
      }
      await reader.cancel().catch(() => {});
    } catch (err: any) {
      console.error("Processing error:", err);
      setError(
        err.message || "An unknown error occurred. Is the backend running?"
      );
    } finally {
      setIsProcessing(false);
    }
  };

  const startProcessing = async () => {
    setProgress(0);
    setProcessedFileCount(0);
    setFiles((prevFiles) =>
      prevFiles.map((f) =>
        f.selected
          ? {
              ...f,
              processed: false,
              classification: undefined,
              qualityMetrics: undefined,
              analysis: undefined,
              error: undefined,
            }
          : f
      )
    );
    // Allow state to update before starting the process
    await new Promise((resolve) => setTimeout(resolve, 50));
    await processFiles(selectedFiles);
  };

  const retryFailedProcessing = async () => {
    const failedFileIds = failedFiles.map((f) => f.id);
    setFiles((prevFiles) =>
      prevFiles.map((f) =>
        failedFileIds.includes(f.id)
          ? {
              ...f,
              processed: false,
              qualityMetrics: undefined,
              analysis: undefined,
              error: undefined,
            }
          : f
      )
    );
    // Allow state to update before starting the process
    await new Promise((resolve) => setTimeout(resolve, 50));
    await processFiles(failedFiles);
  };

  // --- Helper functions for rendering ---

  const getQualityBadge = (score: number) => {
    if (score >= 3)
      return (
        <Badge className="bg-green-500 hover:bg-green-600">Excellent</Badge>
      );
    if (score >= 2)
      return <Badge className="bg-yellow-500 hover:bg-yellow-600">Good</Badge>;
    if (score >= 1)
      return <Badge className="bg-orange-500 hover:bg-orange-600">Fair</Badge>;
    return <Badge variant="destructive">Poor</Badge>;
  };

  const getFileStatus = (file: FileData) => {
    if (file.processed) {
      return file.error ? (
        <AlertCircle className="w-5 h-5 text-red-500" />
      ) : (
        <CheckCircle className="w-5 h-5 text-green-500" />
      );
    }
    if (isProcessing && file.selected) {
      return <Clock className="w-5 h-5 text-blue-500 animate-spin" />;
    }
    return <div className="w-5 h-5 rounded-full border-2 border-gray-300" />;
  };

  const DetailItem = ({
    label,
    value,
  }: {
    label: string;
    value: React.ReactNode;
  }) => {
    if (!value || (Array.isArray(value) && value.length === 0)) return null;
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-1 py-2 border-b">
        <dt className="font-semibold text-gray-600">{label}</dt>
        <dd className="md:col-span-2 text-gray-800">{value}</dd>
      </div>
    );
  };

  // --- Render ---

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Step 2: File Processing</h2>
        <p className="text-gray-600">
          Analyze data quality and classify file types via API
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Processing Control</CardTitle>
          <CardDescription>
            Process {selectedFiles.length} selected files to compute quality
            metrics.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col sm:flex-row items-center gap-4 mb-4">
            <div className="flex items-center gap-4">
              <Button
                onClick={startProcessing}
                disabled={
                  isProcessing ||
                  selectedFiles.length === 0 ||
                  allFilesProcessed
                }
                className="flex items-center gap-2 w-40"
              >
                <Play className="w-4 h-4" />
                {isProcessing
                  ? "Processing..."
                  : allFilesProcessed
                  ? "Completed"
                  : "Start Processing"}
              </Button>
              <Button
                variant="outline"
                onClick={retryFailedProcessing}
                disabled={isProcessing || failedFiles.length === 0}
                className="flex items-center gap-2"
              >
                <RotateCcw className="w-4 h-4" />
                Retry Failed ({failedFiles.length})
              </Button>
            </div>
            <div className="flex-1 w-full">
              <div className="flex justify-between text-sm mb-1">
                <span>Progress</span>
                <span>{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="w-full" />
            </div>
          </div>

          {error && (
            <div className="mt-4 flex items-center gap-2 text-sm text-red-600 bg-red-50 p-3 rounded-md">
              <AlertCircle className="w-4 h-4" />
              <span>Error: {error}</span>
            </div>
          )}

          <div className="text-sm text-gray-600 mt-2">
            Status:{" "}
            {isProcessing
              ? "Processing files on server..."
              : `${processedFileCount} of ${selectedFiles.length} files processed.`}
          </div>
        </CardContent>
      </Card>

      {selectedFiles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Processing Results</CardTitle>
            <CardDescription>
              Data quality metrics and classifications from the server.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">Status</TableHead>
                  <TableHead>File Name</TableHead>
                  <TableHead>Classification</TableHead>
                  <TableHead>Parse Accuracy</TableHead>
                  <TableHead>Domain</TableHead>
                  <TableHead>Sub Domain</TableHead>
                  <TableHead>Intent</TableHead>
                  <TableHead>More Details</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {selectedFiles.map((file) => (
                  <TableRow
                    key={file.id}
                    className={
                      file.processed
                        ? file.error
                          ? "bg-red-50/50"
                          : "bg-green-50/50"
                        : ""
                    }
                  >
                    <TableCell>{getFileStatus(file)}</TableCell>
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-gray-500" />
                        {file.name}
                      </div>
                    </TableCell>
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-gray-500" />
                        {file.classification}
                      </div>
                    </TableCell>
                    <TableCell>
                      {file.error ? (
                        <Badge variant="destructive">Failed</Badge>
                      ) : file.qualityMetrics ? (
                        <div className="flex items-center gap-2">
                          <span>{file.qualityMetrics.parseAccuracy}/3</span>
                          {getQualityBadge(file.qualityMetrics.parseAccuracy)}
                        </div>
                      ) : (
                        <span className="text-gray-400">
                          {isProcessing && file.selected
                            ? "Processing..."
                            : "Pending"}
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      {file.analysis?.domain ? (
                        <Badge variant="outline">{file.analysis.domain}</Badge>
                      ) : (
                        <span className="text-gray-400">
                          {isProcessing && file.selected ? "..." : "Pending"}
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      {file.analysis?.subdomain ? (
                        <Badge variant="outline">
                          {file.analysis.subdomain}
                        </Badge>
                      ) : (
                        <span className="text-gray-400">
                          {isProcessing && file.selected ? "..." : "Pending"}
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="text-xs text-gray-600 max-w-xs">
                      <div className="flex flex-wrap gap-1">
                        {file.analysis?.intents &&
                          (Array.isArray(file.analysis.intents) ? (
                            file.analysis.intents.map((intent, idx) => (
                              <Badge key={idx} variant="default">
                                {intent}
                              </Badge>
                            ))
                          ) : (
                            <Badge variant="default">
                              {file.analysis.intents}
                            </Badge>
                          ))}
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      {file.processed && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setSelectedFileForDetails(file)}
                          className="flex items-center gap-1"
                        >
                          <Eye className="w-4 h-4" /> View
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* --- Details Modal --- */}
      {selectedFileForDetails && selectedFileForDetails.analysis && (
        <Dialog
          open={!!selectedFileForDetails}
          onOpenChange={() => setSelectedFileForDetails(null)}
        >
          <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                {selectedFileForDetails.analysis.file_name ||
                  selectedFileForDetails.name}
              </DialogTitle>
              <DialogDescription>
                Detailed analysis and metadata extracted from the document.
              </DialogDescription>
            </DialogHeader>

            <div className="text-sm">
              <dl>
                {/* Display file-specific error in modal */}
                {selectedFileForDetails.error && (
                  <DetailItem
                    label="Processing Error"
                    value={
                      <div className="text-red-600 p-2 bg-red-50 rounded-md">
                        {selectedFileForDetails.error}
                      </div>
                    }
                  />
                )}
                <DetailItem
                  label="Brief Summary"
                  value={selectedFileForDetails.analysis.brief_summary}
                />
                <DetailItem
                  label="Domain"
                  value={
                    <Badge variant="outline">
                      {selectedFileForDetails.analysis.domain}
                    </Badge>
                  }
                />
                <DetailItem
                  label="Subdomain"
                  value={selectedFileForDetails.analysis.subdomain}
                />
                <DetailItem
                  label="Publishing Authority"
                  value={selectedFileForDetails.analysis.publishing_authority}
                />
                <DetailItem
                  label="Published Date"
                  value={selectedFileForDetails.analysis.published_date}
                />
                <DetailItem
                  label="Period of Reference"
                  value={selectedFileForDetails.analysis.period_of_reference}
                />
                <DetailItem
                  label="Document Size"
                  value={selectedFileForDetails.size}
                />
                <DetailItem
                  label="Intents"
                  value={
                    <div className="flex flex-wrap gap-1">
                      {selectedFileForDetails.analysis.intents &&
                        (Array.isArray(
                          selectedFileForDetails.analysis.intents
                        ) ? (
                          selectedFileForDetails.analysis.intents.map(
                            (intent, idx) => (
                              <Badge key={idx} variant="default">
                                {intent}
                              </Badge>
                            )
                          )
                        ) : (
                          <Badge variant="default">
                            {selectedFileForDetails.analysis.intents}
                          </Badge>
                        ))}
                    </div>
                  }
                />
                {selectedFileForDetails.analysis.extra_fields &&
                  Object.entries(
                    selectedFileForDetails.analysis.extra_fields
                  ).map(([key, value]) => (
                    <DetailItem
                      key={key}
                      label={key
                        .replace(/_/g, " ")
                        .replace(/\b\w/g, (l) => l.toUpperCase())}
                      value={
                        typeof value === "object"
                          ? JSON.stringify(value, null, 2)
                          : String(value)
                      }
                    />
                  ))}
              </dl>
            </div>

            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => setSelectedFileForDetails(null)}
              >
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
