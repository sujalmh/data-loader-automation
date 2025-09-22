"use client";

import type React from "react";
import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  File as FileIcon,
  FileText as FileTextIcon,
  FileSpreadsheet as FileSpreadsheetIcon,
  Image as ImageIcon,
  Upload as UploadIcon,
  Download as DownloadIcon,
  Trash2 as Trash2Icon,
  Loader2 as Loader2Icon,
  ArrowRight as ArrowRightIcon,
  CheckCircle2 as CheckCircle2Icon,
  Link as LinkIcon,
  AlertTriangle as AlertTriangleIcon, // Import for duplicate icon
} from "lucide-react";
import type { FileData } from "@/app/page";

// Extend FileData to include duplicate status
interface ExtendedFileData extends FileData {
  isDuplicate?: boolean;
}

interface FileSelectionProps {
  files: ExtendedFileData[];
  setFiles: (files: ExtendedFileData[]) => void;
  onUploadComplete: () => void;
}

export default function FileSelection({
  files,
  setFiles,
  onUploadComplete,
}: FileSelectionProps) {
  const [urlInput, setUrlInput] = useState<string>("");
  const [bulkSourceUrl, setBulkSourceUrl] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = Array.from(event.target.files || []);
    if (uploadedFiles.length === 0) return;

    const newFiles: ExtendedFileData[] = uploadedFiles.map((file) => ({
      id: `${file.name}-${file.size}-${file.lastModified}`,
      name: file.name,
      path: file.name,
      size: file.size,
      type: file.name.split(".").pop()?.toLowerCase() || "unknown",
      selected: true,
      processed: false,
      uploaded: false,
      isDuplicate: false, // Initialize duplicate status
      file: file,
      sourceUrl: "",
    }));

    setFiles((prevFiles) => {
      const existingIds = new Set(prevFiles.map((f) => f.id));
      const uniqueNewFiles = newFiles.filter((f) => !existingIds.has(f.id));
      return [...prevFiles, ...uniqueNewFiles];
    });
  };

  const handleUrlUpload = async () => {
    if (!urlInput.trim() || isLoading) return;
    const urls = urlInput
      .split(",")
      .map((url) => url.trim())
      .filter(Boolean);
    if (urls.length === 0) return;

    setIsLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const response = await fetch(`${apiUrl}/download-from-urls`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ urls: urls }),
      });

      if (!response.ok) {
        console.error("Backend error:", response.statusText);
        return;
      }
      
      // --- CHANGE START: Define a more specific type for the backend response ---
      const results: Array<{
        status: "success" | "error" | "duplicate";
        url: string;
        // Success fields
        id?: string;
        data?: {
          name: string;
          path: string;
          size: number;
          type: string;
          source_url: string;
          file_base64: string;
        };
        // Error fields
        error?: string;
        // Duplicate fields
        message?: string;
        existing_file_id?: string;
      }> = await response.json();
      // --- CHANGE END ---

      const downloadedFiles: ExtendedFileData[] = [];
      for (const result of results) {
        if (result.status === "success" && result.data && result.id) {
          const byteCharacters = atob(result.data.file_base64);
          const byteNumbers = new Array(byteCharacters.length);
          for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
          }
          const byteArray = new Uint8Array(byteNumbers);
          const blob = new Blob([byteArray], { type: result.data.type });
          const fileObject = new File([blob], result.data.name, { type: result.data.type });

          const fileData: ExtendedFileData = {
            id: result.id,
            name: result.data.name,
            path: result.data.path,
            size: result.data.size,
            type: result.data.type,
            selected: true,
            processed: false,
            uploaded: true,
            isDuplicate: false,
            file: fileObject,
            sourceUrl: new URL(result.url).toString(),
          };
          downloadedFiles.push(fileData);
        
        // --- CHANGE START: Handle the 'duplicate' status from the backend ---
        } else if (result.status === "duplicate" && result.existing_file_id) {
            const urlPath = new URL(result.url).pathname;
            const fileName = urlPath.substring(urlPath.lastIndexOf('/') + 1) || result.url;
            
            const duplicateData: ExtendedFileData = {
                id: result.existing_file_id,
                name: fileName,
                path: 'N/A',
                size: 0,
                type: fileName.split(".").pop()?.toLowerCase() || "unknown",
                selected: false, // Duplicates are not selected
                processed: true, // Mark as processed to prevent action
                uploaded: false,
                isDuplicate: true, // Set the duplicate flag
                file: undefined, // No file object for duplicates
                sourceUrl: result.url,
            };
            downloadedFiles.push(duplicateData);
        }
        // --- CHANGE END ---
      }
      
      setFiles((prevFiles) => {
        const existingIds = new Set(prevFiles.map((f) => f.id));
        const uniqueNewFiles = downloadedFiles.filter((f) => !existingIds.has(f.id));
        return [...prevFiles, ...uniqueNewFiles];
      });
    } catch (error) {
      console.error("Failed to connect to the backend:", error);
    } finally {
      setIsLoading(false);
      setUrlInput("");
    }
  };

  const handleUploadAndProceed = async () => {
    const selectedFiles = files.filter((f) => f.selected && !f.isDuplicate); // Ensure duplicates are not processed
    if (selectedFiles.length === 0) {
      alert("Please select at least one new file to upload.");
      return;
    }

    setIsUploading(true);
    const localFilesToUpload = selectedFiles.filter((f) => f.file && !f.uploaded);

    try {
      let uploadedFileResults: any[] = [];
      let duplicateFileResults: any[] = []; // To store duplicate info from backend

      if (localFilesToUpload.length > 0) {
        const formData = new FormData();
        localFilesToUpload.forEach((fileData) => {
          if (fileData.file) {
            formData.append("files", fileData.file, fileData.name);
          }
        });

        const apiUrl = process.env.NEXT_PUBLIC_API_URL;
        const response = await fetch(`${apiUrl}/upload-files/`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
        }
        
        // --- CHANGE START: Handle the new response structure with 'new_files' and 'duplicates' ---
        const result = await response.json();
        uploadedFileResults = result.new_files || [];
        duplicateFileResults = result.duplicates || [];
        // --- CHANGE END ---
      }

      const serverFileMap = new Map(uploadedFileResults.map((f: { id: string; name: string; path: string }) => [f.name, f]));
      const duplicateNameSet = new Set(duplicateFileResults.map((d: { name: string }) => d.name));
      const alreadyUploadedFiles = new Set(selectedFiles.filter((f) => f.uploaded).map((f) => f.name));

      setFiles((prevFiles) =>
        prevFiles.map((pf) => {
          // --- CHANGE START: Check for duplicates first ---
          if (pf.selected && duplicateNameSet.has(pf.name)) {
            return {
              ...pf,
              isDuplicate: true,
              selected: false, // Deselect the file
              processed: true,
            };
          }
          // --- CHANGE END ---
          
          if (serverFileMap.has(pf.name)) {
            const serverData = serverFileMap.get(pf.name)!;
            return {
              ...pf,
              id: serverData.id,
              path: serverData.path,
              uploaded: true,
            };
          }
          
          if (alreadyUploadedFiles.has(pf.name)) {
            return { ...pf, uploaded: true };
          }
          
          return pf;
        })
      );
      
      onUploadComplete();
    } catch (error) {
      console.error("Failed to upload files:", error);
    } finally {
      setIsUploading(false);
    }
  };

  // --- No changes to handlers below this line, only to JSX ---

  const handleSourceUrlChange = (fileId: string, url: string) => {
    setFiles((prevFiles) =>
      prevFiles.map((file) =>
        file.id === fileId ? { ...file, sourceUrl: url } : file
      )
    );
  };

  const handleApplyBulkSourceUrl = () => {
    if (!bulkSourceUrl.trim()) return;
    setFiles((prevFiles) =>
      prevFiles.map((file) =>
        !file.uploaded ? { ...file, sourceUrl: bulkSourceUrl } : file
      )
    );
  };

  const handleUrlInputKeyDown = (
    event: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (event.key === "Enter") {
      event.preventDefault();
      handleUrlUpload();
    }
  };

  const handleDeleteFile = (fileId: string) => {
    setFiles((prevFiles) => prevFiles.filter((file) => file.id !== fileId));
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const toggleFileSelection = (fileId: string) => {
    setFiles(
      files.map((file) =>
        file.id === fileId ? { ...file, selected: !file.selected } : file
      )
    );
  };

  const toggleAllFiles = (checked: boolean) => {
    setFiles(files.map((file) => ({ ...file, selected: checked })));
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const getFileIcon = (type: string) => {
    switch (type) {
      case "csv":
      case "xlsx":
      case "xls":
        return <FileSpreadsheetIcon className="h-4 w-4 shrink-0" />;
      case "pdf":
      case "docx":
      case "doc":
      case "txt":
        return <FileTextIcon className="h-4 w-4 shrink-0" />;
      case "png":
      case "jpg":
      case "jpeg":
      case "gif":
        return <ImageIcon className="h-4 w-4 shrink-0" />;
      default:
        return <FileIcon className="h-4 w-4 shrink-0" />;
    }
  };

  const selectedCount = files.filter((f) => f.selected).length;
  const allFilesUploaded = files.length > 0 && files.every((file) => file.uploaded || file.isDuplicate);

  return (
    <div className="space-y-6">
      {/* ... (No changes to the header and upload cards) ... */}
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Step 1: Add Your Files</h2>
        <p className="text-gray-600">
          Upload files from your device or enter URLs to download them.
        </p>
      </div>

      <input ref={fileInputRef} type="file" multiple className="hidden" onChange={handleFileUpload} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><UploadIcon className="h-5 w-5" /> Upload from Device</CardTitle>
            <CardDescription>Click the button to select one or more files.</CardDescription>
          </CardHeader>
          <CardContent>
            <Button onClick={handleUploadClick} className="w-full">
              <UploadIcon className="mr-2 h-4 w-4" /> Browse Files
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><DownloadIcon className="h-5 w-5" /> Download from URLs</CardTitle>
            <CardDescription>Paste URLs and press Enter or click the button.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Input
                placeholder="https://.../file.pdf, https://.../img.png"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                onKeyDown={handleUrlInputKeyDown}
                disabled={isLoading}
                className="flex-grow"
              />
              <Button onClick={handleUrlUpload} disabled={isLoading || !urlInput.trim()} className="shrink-0">
                {isLoading ? <Loader2Icon className="h-4 w-4 animate-spin" /> : <DownloadIcon className="h-4 w-4" />}
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>

      {files.length > 0 && (
        <Card>
            <CardHeader>
                <CardTitle>File Queue</CardTitle>
                <CardDescription>{files.length} file(s) added • {selectedCount} selected</CardDescription>
            </CardHeader>
            <CardContent>
                {/* ... (No changes to the "Select All" / "Clear All" section) ... */}
                <div className="mb-4 flex items-center justify-between gap-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="select-all"
                      checked={files.length > 0 && selectedCount === files.length}
                      onCheckedChange={(v) => toggleAllFiles(Boolean(v))}
                    />
                    <label htmlFor="select-all" className="text-sm font-medium">Select All</label>
                  </div>
                  <Button variant="outline" size="sm" onClick={() => setFiles([])}>Clear All</Button>
                </div>
                {/* ... (No changes to the Bulk URL section) ... */}
                {files.some((f) => !f.uploaded) && (
                  <div className="mb-4 space-y-2 rounded-lg border bg-muted/40 p-3">
                    <label htmlFor="bulk-source-url" className="text-sm font-medium">Bulk Add Source URL</label>
                    <div className="flex gap-2">
                      <div className="relative flex-grow">
                        <LinkIcon className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
                        <Input
                          id="bulk-source-url"
                          placeholder="Enter one URL for all non-uploaded files"
                          value={bulkSourceUrl}
                          onChange={(e) => setBulkSourceUrl(e.target.value)}
                          className="pl-9"
                        />
                      </div>
                      <Button onClick={handleApplyBulkSourceUrl} disabled={!bulkSourceUrl.trim()}>Apply to All</Button>
                    </div>
                  </div>
                )}
                <div className="max-h-96 space-y-3 overflow-y-auto pr-2">
                    {files.map((file) => (
                        <div key={file.id} className="rounded-lg border p-3 hover:bg-gray-50/50">
                            <div className="flex items-center gap-3">
                                <Checkbox
                                    checked={file.selected}
                                    onCheckedChange={() => toggleFileSelection(file.id)}
                                    // --- CHANGE START: Disable checkbox if file is a duplicate ---
                                    disabled={file.isDuplicate}
                                    // --- CHANGE END ---
                                />
                                {getFileIcon(file.type)}
                                <div className="flex-1 overflow-hidden">
                                    <p className="truncate font-medium" title={file.name}>{file.name}</p>
                                    <p className="text-sm text-gray-500">{formatFileSize(file.size)}</p>
                                </div>
                                
                                {/* --- CHANGE START: Conditionally render Duplicate / Uploaded badge --- */}
                                {file.isDuplicate ? (
                                    <Badge variant="destructive" className="flex shrink-0 items-center gap-1">
                                        <AlertTriangleIcon className="h-3 w-3" />
                                        Duplicate
                                    </Badge>
                                ) : file.uploaded && (
                                    <Badge variant="outline" className="flex shrink-0 items-center gap-1 border-green-500 bg-green-50 text-green-700">
                                        <CheckCircle2Icon className="h-3 w-3" />
                                        Uploaded
                                    </Badge>
                                )}
                                {/* --- CHANGE END --- */}

                                <Badge variant="secondary" className="shrink-0">{file.type.toUpperCase()}</Badge>
                                <Button variant="ghost" size="icon" onClick={() => handleDeleteFile(file.id)} className="shrink-0">
                                    <Trash2Icon className="h-4 w-4 text-gray-500 hover:text-red-500" />
                                </Button>
                            </div>
                            
                            {/* --- CHANGE START: Disable source URL input for duplicates --- */}
                            {!file.uploaded && !file.isDuplicate ? (
                                <div className="relative mt-2 pl-9">
                                    <LinkIcon className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
                                    <Input
                                        type="url"
                                        placeholder="Add Source URL (optional)"
                                        value={file.sourceUrl || ""}
                                        onChange={(e) => handleSourceUrlChange(file.id, e.target.value)}
                                        className="h-9 pl-8 text-sm"
                                    />
                                </div>
                            ) : file.sourceUrl ? (
                            // --- CHANGE END ---
                                <div className="mt-2 pl-9">
                                    <p className="truncate text-sm text-gray-500" title={file.sourceUrl}>
                                        Source: <a href={file.sourceUrl} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">{file.sourceUrl}</a>
                                    </p>
                                </div>
                            ) : null}
                        </div>
                    ))}
                </div>
            </CardContent>
        </Card>
      )}
      {files.length > 0 && (
        <div className="flex justify-end pt-4">
            <Button
                size="lg"
                onClick={handleUploadAndProceed}
                disabled={selectedCount === 0 || isUploading || allFilesUploaded}
            >
                {allFilesUploaded ? ( <CheckCircle2Icon className="mr-2 h-4 w-4" /> )
                : isUploading ? ( <Loader2Icon className="mr-2 h-4 w-4 animate-spin" /> )
                : ( <ArrowRightIcon className="mr-2 h-4 w-4" /> )}
                {allFilesUploaded ? "Ready to Continue" : isUploading ? "Uploading..." : `Upload ${selectedCount} File(s) & Continue`}
            </Button>
        </div>
      )}
    </div>
  );
}