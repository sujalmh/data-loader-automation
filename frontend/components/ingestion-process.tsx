"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Play, ChevronDown, ChevronRight, Database, FileText, Table, CheckCircle, AlertTriangle, RefreshCw } from "lucide-react"
import type { FileData, DatabaseConfig, FileIngestionResult } from "@/app/page"

// ## Type Definitions
// These types define the shape of the data used throughout the component.

type ColumnSchema = {
    name: string;
    type: string;
    primary?: boolean;
};

type TableDetails = {
    tableName: string;
    schema_details: ColumnSchema[];
    rowsInserted: number;
    sqlCommands: string[];
    fileSelectorPrompt?: string;
};

type StructuredIngestionDetails = {
    type: "structured";
    tables: TableDetails[];
};

type UnstructuredIngestionDetails = {
    type: "unstructured";
    collection: string;
    chunksCreated: number;
    embeddingsGenerated: number;
    chunkingMethod: string;
    embeddingModel: string;
};

type IngestionDetails = (StructuredIngestionDetails | UnstructuredIngestionDetails) & {
    startTime: string;
    endTime: string;
};

interface IngestionProcessProps {
    files: FileData[]
    setFiles: (files: (prevFiles: FileData[]) => FileData[]) => void
    progress: number
    setProgress: (progress: number) => void
}

export default function IngestionProcess({
    files,
    setFiles,
    progress,
    setProgress,
}: IngestionProcessProps) {
    const [isIngesting, setIsIngesting] = useState(false)
    const [apiError, setApiError] = useState<string | null>(null);
    const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set())

    const selectedFiles = files.filter((f) => f.selected && f.processed)

    /**
     * Processes ingestion details by adding timestamps.
     * This is called when a final success/failure message for a file is received.
     */
    const processIngestionDetails = (details: IngestionDetails | null): IngestionDetails | null => {
        if (!details) return null;
        const now = new Date().toISOString();
        // Since the stream might send details for structured and unstructured in a single payload,
        // we just add the timestamps to the top-level object.
        return {
            ...details,
            startTime: now,
            endTime: now,
        };
    };

   /**
     * Handles the API call and processes the streaming response for file ingestion.
     */
    const handleIngestion = async (filesToProcess: FileData[]) => {
        if (filesToProcess.length === 0) return;

        setIsIngesting(true);
        setApiError(null);
        setProgress(0);

        // Set initial status to "pending" for files being processed
        setFiles((prevFiles) =>
            prevFiles.map((f) =>
                filesToProcess.some((p) => p.id === f.id)
                    ? { ...f, ingestionStatus: "pending" as const, ingestionDetails: null, error: undefined }
                    : f
            )
        );

        const formData = new FormData();
        const fileDetails = filesToProcess.map(fileData => {
            const { file, ...details } = fileData;
            if (file) {
                formData.append("files", file);
            }
            return details;
        });

        formData.append("file_details", JSON.stringify(fileDetails));

        try {
            const apiUrl = process.env.NEXT_PUBLIC_API_URL;
            const response = await fetch(`${apiUrl}/ingest/`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok || !response.body) {
                const errorData = await response.json().catch(() => ({ detail: "Ingestion request failed with no body." }));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            // Set up to read the response stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            // Loop to read the stream until it's done
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // Decode the chunk and add it to our buffer
                buffer += decoder.decode(value, { stream: true });

                // Split buffer by the SSE message delimiter "\n\n"
                const messages = buffer.split("\n\n");
                
                // The last part might be an incomplete message, so we keep it in the buffer
                buffer = messages.pop() || "";

                for (const message of messages) {
                    if (message.startsWith("data: ")) {
                        try {
                            const jsonString = message.substring(6);
                            const result: FileIngestionResult = JSON.parse(jsonString);

                            // Update overall progress from the stream data
                            setProgress(result.progress);

                            // Update the specific file's status in the UI
                            setFiles((prevFiles) =>
                                prevFiles.map((f) =>
                                    f.id === result.fileId
                                        ? {
                                            ...f,
                                            ingestionStatus: result.status,
                                            ingestionDetails: processIngestionDetails(result.ingestionDetails),
                                            error: result.error || undefined,
                                        }
                                        : f
                                )
                            );

                        } catch (e) {
                            console.error("Failed to parse stream message JSON:", e, message);
                        }
                    }
                }
            }
        } catch (error) {
            console.error("Ingestion stream error:", error);
            const errorMessage = error instanceof Error ? error.message : "An unknown error occurred.";
            setApiError(errorMessage);

            // Mark any remaining pending files as failed
            setFiles((prevFiles) =>
                prevFiles.map((f) =>
                    f.ingestionStatus === "pending"
                        ? { ...f, ingestionStatus: "failed" as const, error: "API connection failed" }
                        : f
                )
            );
        } finally {
            setIsIngesting(false);
            setProgress(100); // Ensure progress bar completes
        }
    }


    // --- Helper functions and JSX remain largely the same, but with updated progress logic ---

    const startIngestion = async () => {
        const filesToIngest = files.filter((f) => f.selected && f.processed);
        await handleIngestion(filesToIngest);
    }

    const retryFailedIngestion = async () => {
        const failedFiles = files.filter((f) => f.ingestionStatus === "failed");
        await handleIngestion(failedFiles);
    }

    const toggleFileExpansion = (fileId: string) => {
        const newExpanded = new Set(expandedFiles)
        if (newExpanded.has(fileId)) {
            newExpanded.delete(fileId)
        } else {
            newExpanded.add(fileId)
        }
        setExpandedFiles(newExpanded)
    }

    const getStatusBadge = (status?: string) => {
        switch (status) {
            case "success":
                return <Badge className="bg-green-500 hover:bg-green-600">✓ Success</Badge>
            case "failed":
                return <Badge variant="destructive">✗ Failed</Badge>
            case "pending":
                return <Badge className="bg-yellow-500 text-white animate-pulse">⏳ Processing</Badge>
            default:
                return <Badge variant="outline">Waiting</Badge>
        }
    }

    const renderIngestionDetails = (details: IngestionDetails) => {
        // This function remains the same as it correctly renders the details object
        return (
            <div className="space-y-4">
                {details.type === "structured" && (
                     <div className="space-y-4">
                        {details.tables.map((table, tableIndex) => (
                            <div key={tableIndex} className="space-y-3 border-l-2 border-blue-500 pl-4">
                                <div className="flex items-center gap-2">
                                    <Table className="w-4 h-4 text-blue-500" />
                                    <h4 className="font-semibold">Table: {table.tableName}</h4>
                                </div>
                                <div className="grid grid-cols-2 gap-4 text-sm">
                                    <div><strong>Rows Inserted:</strong> {table.rowsInserted.toLocaleString()}</div>
                                    <div>
                                         <strong>Schema Details:</strong>
                                         <div className="mt-1 p-2 bg-gray-50 rounded text-xs font-mono max-h-40 overflow-y-auto">
                                            {table.schema_details.map((schema, s_index) => ( <div key={s_index}> {schema.name} ({schema.type}) </div> ))}
                                         </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
                {details.type === "unstructured" && (
                    <div className="space-y-3 border-l-2 border-purple-500 pl-4">
                        <div className="flex items-center gap-2">
                            <Database className="w-4 h-4 text-purple-500" />
                            <h4 className="font-semibold">Vector Ingestion Details</h4>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-2 text-sm">
                            <div><strong>Collection:</strong> {details.collection}</div>
                            <div><strong>Chunks Created:</strong> {details.chunksCreated.toLocaleString()}</div>
                            <div><strong>Embeddings:</strong> {details.embeddingsGenerated.toLocaleString()}</div>
                            <div><strong>Chunking Method:</strong> {details.chunkingMethod}</div>
                            <div className="md:col-span-2"><strong>Embedding Model:</strong> {details.embeddingModel}</div>
                        </div>
                    </div>
                )}
                {details.startTime && details.endTime && (
                     <div className="text-xs text-gray-500 pt-2 border-t mt-4">
                        <div>Started: {new Date(details.startTime).toLocaleString()}</div>
                        <div>Completed: {new Date(details.endTime).toLocaleString()}</div>
                    </div>
                )}
            </div>
        );
    };

    const successCount = selectedFiles.filter((f) => f.ingestionStatus === "success").length
    const failedCount = selectedFiles.filter((f) => f.ingestionStatus === "failed").length
    const pendingCount = selectedFiles.filter((f) => f.ingestionStatus === "pending").length
    
    // The visual progress is now driven directly by the 'progress' state from the stream
    const progressValue = isIngesting ? progress : (successCount + failedCount) / (selectedFiles.length || 1) * 100;

    return (
        <div className="space-y-6">
            <div className="text-center">
                <h2 className="text-2xl font-bold mb-2">Step 5: Ingestion Process</h2>
                <p className="text-gray-600">Ingest selected files into configured databases</p>
            </div>

            {/* Ingestion Control Card */}
            <Card>
                <CardHeader>
                    <CardTitle>Ingestion Control</CardTitle>
                    <CardDescription>Ingest {selectedFiles.length} selected files into databases</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center gap-4 mb-4">
                        <Button
                            onClick={startIngestion}
                            disabled={isIngesting || selectedFiles.length === 0}
                            className="flex items-center gap-2"
                        >
                            <Play className="w-4 h-4" />
                            {isIngesting ? "Ingesting..." : "Start Ingestion"}
                        </Button>
                        
                        {failedCount > 0 && !isIngesting && (
                            <Button
                                onClick={retryFailedIngestion}
                                disabled={isIngesting}
                                variant="outline"
                                className="flex items-center gap-2"
                            >
                                <RefreshCw className="w-4 h-4" />
                                Retry {failedCount} Failed
                            </Button>
                        )}

                        <div className="flex-1">
                            <div className="flex justify-between text-sm mb-1">
                                <span>Progress</span>
                                <span>{Math.round(progressValue)}%</span>
                            </div>
                            <Progress value={progressValue} className="w-full" />
                        </div>
                    </div>

                    {apiError && (
                        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2 text-red-800">
                            <AlertTriangle className="w-4 h-4" />
                            <span className="text-sm font-medium">Error: {apiError}</span>
                        </div>
                    )}

                    <div className="grid grid-cols-3 gap-4 mt-4">
                        <div className="text-center p-2 border rounded-lg"><div className="text-2xl font-bold text-green-600">{successCount}</div><div className="text-sm text-gray-600">Successful</div></div>
                        <div className="text-center p-2 border rounded-lg"><div className="text-2xl font-bold text-red-600">{failedCount}</div><div className="text-sm text-gray-600">Failed</div></div>
                        <div className="text-center p-2 border rounded-lg"><div className="text-2xl font-bold text-yellow-600">{pendingCount}</div><div className="text-sm text-gray-600">Processing</div></div>
                    </div>
                </CardContent>
            </Card>

            {/* Ingestion Results Card */}
            <Card>
                <CardHeader>
                    <CardTitle>Ingestion Results</CardTitle>
                    <CardDescription>Detailed results for each file ingestion</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        {selectedFiles.map((file) => (
                            <Collapsible key={file.id} open={expandedFiles.has(file.id)} onOpenChange={() => toggleFileExpansion(file.id)}>
                                <div className="border rounded-lg p-4">
                                    <CollapsibleTrigger className="flex items-center justify-between w-full text-left">
                                        <div className="flex items-center gap-3">
                                            {expandedFiles.has(file.id) ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                                            <FileText className="w-4 h-4" />
                                            <span className="font-medium">{file.name}</span>
                                            <Badge variant="outline">{file.classification}</Badge>
                                            {getStatusBadge(file.ingestionStatus)}
                                        </div>
                                    </CollapsibleTrigger>
                                    <CollapsibleContent className="mt-4 pt-4 border-t pl-8">
                                        {file.ingestionStatus === 'failed' && file.error && (
                                            <div className="p-3 bg-red-50 text-red-700 rounded-md text-sm">
                                                <strong>Error:</strong> {file.error}
                                            </div>
                                        )}
                                        {file.ingestionDetails && renderIngestionDetails(file.ingestionDetails as IngestionDetails)}
                                    </CollapsibleContent>
                                </div>
                            </Collapsible>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* Completion Message */}
            {!isIngesting && Math.round(progress) === 100 && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center gap-2 text-green-800">
                        <CheckCircle className="w-5 h-5" />
                        <span className="font-medium">Ingestion Complete!</span>
                    </div>
                    <p className="text-sm text-green-700 mt-1">
                        {successCount} files ingested successfully. {failedCount > 0 && `${failedCount} files failed.`}
                    </p>
                </div>
            )}
        </div>
    );
}






