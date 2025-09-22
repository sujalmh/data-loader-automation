"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Download, FileText, ChevronDown, ChevronRight, Database, Table, BarChart3 } from "lucide-react"
// Assuming FileData and DatabaseConfig are defined elsewhere, as they were in the original file.
import type { FileData, DatabaseConfig } from "@/app/page"

// NOTE: These types are based on the new definitions provided.
// They should ideally be shared with the backend to ensure consistency.
export type ColumnSchema = {
    name: string;
    type: string;
    primary?: boolean;
};

export type TableDetails = {
    tableName: string;
    schema_details: ColumnSchema[];
    rowsInserted: number;
    sqlCommands: string[];
    fileSelectorPrompt?: string;
};

export type StructuredIngestionDetails = {
    type: "structured";
    tables: TableDetails[];
};

export type UnstructuredIngestionDetails = {
    type: "unstructured";
    collection: string;
    chunksCreated: number;
    embeddingsGenerated: number;
    chunkingMethod: string;
    embeddingModel: string;
};

export type IngestionDetails = (StructuredIngestionDetails | UnstructuredIngestionDetails) & {
    startTime: string;
    endTime: string;
};

// This interface adapts the component to the new API response structure.
interface FileDataWithDetails extends Omit<FileData, 'ingestionDetails' | 'error'> {
    ingestionDetails?: IngestionDetails | IngestionDetails[] | { [key: string]: IngestionDetails } | null;
    error?: string | null;
}

interface SummaryViewProps {
    files: FileDataWithDetails[]
    databaseConfig?: DatabaseConfig
}

// Helper component to render details for a single ingestion event.
const IngestionDetailView = ({ details }: { details: IngestionDetails }) => {
    if (details.type === "structured") {
        return (
            <div className="space-y-4">
                {details.tables.map((table, index) => (
                    <div key={index} className="p-3 border rounded space-y-2 bg-gray-50/50">
                        <div className="flex items-center gap-2 mb-2">
                            <Table className="w-4 h-4 text-blue-500" />
                            <span className="font-medium">Table: {table.tableName}</span>
                        </div>
                        <div className="flex items-center gap-2 mb-2">
                                <Table className="w-4 h-4 text-blue-500" />
                                <span className="font-medium">File Selector Prompt: {table.fileSelectorPrompt}</span>
                            </div>
                        
                        <div>
                            <strong>Rows Inserted:</strong> {table.rowsInserted.toLocaleString()}
                        </div>
                        <div>
                            <strong>Schema Details:</strong>
                            <div className="mt-2 p-3 bg-white rounded text-xs font-mono max-h-40 overflow-y-auto border">
                                {table.schema_details.map((schema, s_index) => (
                                    <div key={s_index}>
                                        {schema.name} ({schema.type}) {schema.primary && <Badge variant="secondary" className="ml-2">PK</Badge>}
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    if (details.type === "unstructured") {
        return (
            <div className="p-3 border rounded space-y-2 text-sm bg-gray-50/50">
                <div className="flex items-center gap-2 mb-2">
                    <Database className="w-4 h-4 text-purple-500" />
                    <span className="font-medium">Vector Ingestion</span>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                    <div>
                        <strong>Collection:</strong> {details.collection}
                    </div>
                    <div>
                        <strong>Chunks:</strong> {details.chunksCreated.toLocaleString()}
                    </div>
                    <div>
                        <strong>Embeddings:</strong> {details.embeddingsGenerated.toLocaleString()}
                    </div>
                    <div className="col-span-2">
                        <strong>Model:</strong> {details.embeddingModel}
                    </div>
                </div>
            </div>
        );
    }

    return null;
};


export default function SummaryView({ files, databaseConfig }: SummaryViewProps) {
    const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set())

    const processedFiles = files.filter((f) => f.processed)
    const selectedFiles = files.filter((f) => f.selected && f.processed)
    const successfulIngestions = selectedFiles.filter((f) => f.ingestionStatus === "success")
    const failedIngestions = selectedFiles.filter((f) => f.ingestionStatus === "failed")

    // Filter files based on their classification for the summary view.
    const structuredFiles = successfulIngestions.filter((f) => f.classification === "Structured")
    const unstructuredFiles = successfulIngestions.filter((f) => f.classification === "Unstructured")

    const toggleFileExpansion = (fileId: string) => {
        const newExpanded = new Set(expandedFiles)
        if (newExpanded.has(fileId)) {
            newExpanded.delete(fileId)
        } else {
            newExpanded.add(fileId)
        }
        setExpandedFiles(newExpanded)
    }

    const exportReport = (format: "pdf" | "json") => {
        const reportData = {
            summary: {
                totalFiles: processedFiles.length,
                selectedFiles: selectedFiles.length,
                successfulIngestions: successfulIngestions.length,
                failedIngestions: failedIngestions.length,
                structuredFiles: structuredFiles.length,
                unstructuredFiles: unstructuredFiles.length,
            },
            files: selectedFiles.map((file) => ({
                name: file.name,
                classification: file.classification,
                qualityMetrics: file.qualityMetrics,
                ingestionStatus: file.ingestionStatus,
                ingestionDetails: file.ingestionDetails,
                error: file.error,
            })),
            timestamp: new Date().toISOString(),
        }

        if (format === "json") {
            const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: "application/json" })
            const url = URL.createObjectURL(blob)
            const a = document.createElement("a")
            a.href = url
            a.download = `data-loader-report-${new Date().toISOString().split("T")[0]}.json`
            a.click()
            URL.revokeObjectURL(url) // Clean up the object URL
        } else {
            // A proper implementation would use a library like jsPDF.
            const modal = document.createElement('div');
            modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 1000;';
            modal.innerHTML = `
                <div style="background: white; padding: 2rem; border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
                    <p style="margin-bottom: 1rem;">PDF report generation would be implemented with a library like jsPDF.</p>
                    <button id="close-modal-btn" style="padding: 0.5rem 1rem; border: 1px solid #ccc; border-radius: 0.25rem; cursor: pointer;">Close</button>
                </div>
            `;
            document.body.appendChild(modal);
            document.getElementById('close-modal-btn')?.addEventListener('click', () => {
                document.body.removeChild(modal);
            });
        }
    }
    
    // Helper function to normalize ingestionDetails into an array
    const getIngestionDetailsAsArray = (details: FileDataWithDetails['ingestionDetails']): IngestionDetails[] => {
        if (!details) return [];
        if (Array.isArray(details)) return details;
        // If it's an object (like the console output), convert its values to an array.
        if (typeof details === 'object' && details !== null) return Object.values(details);
        return [];
    }

    return (
        <div className="space-y-6">
            <div className="text-center">
                <h2 className="text-2xl font-bold mb-2">Step 6: Summary & Report</h2>
                <p className="text-gray-600">Comprehensive overview of the data loading process</p>
            </div>

            {/* Overall Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card>
                    <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold text-blue-600">{processedFiles.length}</div>
                        <div className="text-sm text-gray-600">Files Processed</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold text-green-600">{successfulIngestions.length}</div>
                        <div className="text-sm text-gray-600">Successful Ingestions</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold text-red-600">{failedIngestions.length}</div>
                        <div className="text-sm text-gray-600">Failed Ingestions</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardContent className="p-4 text-center">
                        <div className="text-2xl font-bold text-purple-600">
                            {Math.round((successfulIngestions.length / (selectedFiles.length || 1)) * 100)}%
                        </div>
                        <div className="text-sm text-gray-600">Success Rate</div>
                    </CardContent>
                </Card>
            </div>

            {/* Database Distribution */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="w-5 h-5" />
                        Data Distribution by Type
                    </CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="text-center p-4 border rounded-lg">
                            <Database className="w-8 h-8 text-blue-500 mx-auto mb-2" />
                            <div className="text-xl font-bold">{structuredFiles.length}</div>
                            <div className="text-sm text-gray-600">Structured Files</div>
                        </div>
                        <div className="text-center p-4 border rounded-lg">
                            <FileText className="w-8 h-8 text-purple-500 mx-auto mb-2" />
                            <div className="text-xl font-bold">{unstructuredFiles.length}</div>
                            <div className="text-sm text-gray-600">Unstructured Files</div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Detailed File Summary */}
            <Card>
                <CardHeader>
                    <CardTitle>Detailed File Summary</CardTitle>
                    <CardDescription>Comprehensive analysis and insights for each processed file</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        {selectedFiles.map((file) => (
                            <Collapsible key={file.id}>
                                <div className="border rounded-lg p-4">
                                    <CollapsibleTrigger
                                        className="flex items-center justify-between w-full text-left"
                                        onClick={() => toggleFileExpansion(file.id)}
                                    >
                                        <div className="flex items-center gap-3 flex-wrap">
                                            {expandedFiles.has(file.id) ? (
                                                <ChevronDown className="w-4 h-4" />
                                            ) : (
                                                <ChevronRight className="w-4 h-4" />
                                            )}
                                            <FileText className="w-4 h-4" />
                                            <span className="font-medium">{file.name}</span>
                                            <Badge variant="outline">{file.classification}</Badge>
                                            {file.ingestionStatus === "success" ? (
                                                <Badge className="bg-green-100 text-green-800">✓ Success</Badge>
                                            ) : (
                                                <Badge variant="destructive">✗ Failed</Badge>
                                            )}
                                        </div>
                                    </CollapsibleTrigger>

                                    <CollapsibleContent className="mt-4">
                                        <div className="space-y-4 pl-7">
                                            {/* Quality Metrics */}
                                            {file.qualityMetrics && (
                                                <div>
                                                    <h4 className="font-semibold mb-2">Quality Assessment</h4>
                                                    <div className="grid grid-cols-3 gap-4 text-sm">
                                                        <div className="p-2 border rounded">
                                                            <div className="font-medium">Parse Accuracy</div>
                                                            <div>{file.qualityMetrics.parseAccuracy}/3</div>
                                                        </div>
                                                        <div className="p-2 border rounded">
                                                            <div className="font-medium">Completeness</div>
                                                            <div>{file.qualityMetrics.completeness}/3</div>
                                                        </div>
                                                        <div className="p-2 border rounded">
                                                            <div className="font-medium">Complexity</div>
                                                            <div>{file.qualityMetrics.complexity}/3</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}

                                            {/* Ingestion Details (API Results) */}
                                            {file.ingestionStatus === "success" && (
                                                <div>
                                                    <h4 className="font-semibold mb-2">Ingestion Summary</h4>
                                                    <div className="space-y-3">
                                                    {
                                                        getIngestionDetailsAsArray(file.ingestionDetails)
                                                        .map((details, index) => (
                                                            <IngestionDetailView key={index} details={details} />
                                                        ))
                                                    }
                                                    </div>
                                                </div>
                                            )}

                                            {/* Ingestion Failure Message */}
                                            {file.ingestionStatus === "failed" && file.error && (
                                                <div className="p-3 bg-red-50 text-red-700 rounded-md text-sm">
                                                    <strong>Error:</strong> {file.error}
                                                </div>
                                            )}
                                        </div>
                                    </CollapsibleContent>
                                </div>
                            </Collapsible>
                        ))}
                    </div>
                </CardContent>
            </Card>

            {/* Export Options */}
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <Download className="w-5 h-5" />
                        Export Report
                    </CardTitle>
                    <CardDescription>Generate a comprehensive report of the data loading process</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center gap-4">
                        <Button onClick={() => exportReport("pdf")} className="flex items-center gap-2">
                            <Download className="w-4 h-4" />
                            Export as PDF
                        </Button>
                        <Button variant="outline" onClick={() => exportReport("json")} className="flex items-center gap-2">
                            <Download className="w-4 h-4" />
                            Export as JSON
                        </Button>
                    </div>
                    <div className="mt-4 text-sm text-gray-600">
                        <p>The report includes:</p>
                        <ul className="list-disc list-inside mt-2 space-y-1">
                            <li>Complete processing statistics and success rates</li>
                            <li>Detailed quality metrics for each file</li>
                            <li>Database ingestion logs and schema information</li>
                            <li>Configuration details and connection information</li>
                        </ul>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
