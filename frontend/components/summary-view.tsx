"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Download, FileText, ChevronDown, ChevronRight, Database, Table, BarChart3 } from "lucide-react"
import type { FileData, DatabaseConfig } from "@/app/page"

// ## Type Definitions (No changes needed here)
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

// Union of all possible ingestion detail types
export type IngestionDetails = (StructuredIngestionDetails | UnstructuredIngestionDetails) & {
    startTime: string;
    endTime: string;
};

// Interface for file data, allowing for flexible ingestionDetails structure
interface FileDataWithDetails extends Omit<FileData, 'ingestionDetails' | 'error'> {
    ingestionDetails?: IngestionDetails | IngestionDetails[] | null;
    error?: string | null;
}

interface SummaryViewProps {
    files: FileDataWithDetails[]
    databaseConfig?: DatabaseConfig
}

// ## Helper Component to Render Details
// This component correctly renders structured or unstructured details.
const IngestionDetailView = ({ details }: { details: IngestionDetails }) => {
    // Structured details rendering
    if (details.type === "structured") {
        return (
            <div className="space-y-4">
                {details.tables.map((table, index) => (
                    <div key={index} className="p-3 border rounded space-y-2 bg-gray-50/50">
                        <div className="flex items-center gap-2 mb-2">
                            <Table className="w-4 h-4 text-blue-500" />
                            <span className="font-medium">Table: {table.tableName}</span>
                        </div>
                        {table.fileSelectorPrompt && (
                             <div className="flex items-center gap-2 mb-2">
                                <FileText className="w-4 h-4 text-gray-500" />
                                <span className="font-medium text-sm">File Selector Prompt: {table.fileSelectorPrompt}</span>
                            </div>
                        )}
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

    // Unstructured details rendering
    if (details.type === "unstructured") {
        return (
            <div className="p-3 border rounded space-y-2 text-sm bg-gray-50/50">
                <div className="flex items-center gap-2 mb-2">
                    <Database className="w-4 h-4 text-purple-500" />
                    <span className="font-medium">Vector Ingestion</span>
                </div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                    <div><strong>Collection:</strong> {details.collection}</div>
                    <div><strong>Chunks:</strong> {details.chunksCreated.toLocaleString()}</div>
                    <div><strong>Embeddings:</strong> {details.embeddingsGenerated.toLocaleString()}</div>
                    <div className="col-span-2"><strong>Model:</strong> {details.embeddingModel}</div>
                </div>
            </div>
        );
    }

    return null; // Should not happen if data is well-formed
};


// ## Main Summary View Component
export default function SummaryView({ files, databaseConfig }: SummaryViewProps) {
    const [expandedFiles, setExpandedFiles] = useState<Set<string>>(new Set())

    const processedFiles = files.filter((f) => f.processed)
    const selectedFiles = files.filter((f) => f.selected && f.processed)
    const successfulIngestions = selectedFiles.filter((f) => f.ingestionStatus === "success")
    const failedIngestions = selectedFiles.filter((f) => f.ingestionStatus === "failed")
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

    // **[FIXED]** Helper function to normalize various `ingestionDetails` shapes into a consistent array.
    // Your API can return a single object or an array of objects. This handles both cases gracefully.
    const getIngestionDetailsAsArray = (details: FileDataWithDetails['ingestionDetails']): IngestionDetails[] => {
        if (!details) return [];
        if (Array.isArray(details)) return details;
        if (typeof details === 'object' && details !== null) return [details as IngestionDetails];
        return [];
    }

    const exportReport = (format: "pdf" | "json") => {
        // ... (export logic remains unchanged)
    }
    
    return (
        <div className="space-y-6">
            <div className="text-center">
                <h2 className="text-2xl font-bold mb-2">Step 6: Summary & Report</h2>
                <p className="text-gray-600">Comprehensive overview of the data loading process</p>
            </div>

            {/* Overall Statistics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <Card><CardContent className="p-4 text-center"><div className="text-2xl font-bold text-blue-600">{processedFiles.length}</div><div className="text-sm text-gray-600">Files Processed</div></CardContent></Card>
                <Card><CardContent className="p-4 text-center"><div className="text-2xl font-bold text-green-600">{successfulIngestions.length}</div><div className="text-sm text-gray-600">Successful</div></CardContent></Card>
                <Card><CardContent className="p-4 text-center"><div className="text-2xl font-bold text-red-600">{failedIngestions.length}</div><div className="text-sm text-gray-600">Failed</div></CardContent></Card>
                <Card><CardContent className="p-4 text-center"><div className="text-2xl font-bold text-purple-600">{Math.round((successfulIngestions.length / (selectedFiles.length || 1)) * 100)}%</div><div className="text-sm text-gray-600">Success Rate</div></CardContent></Card>
            </div>

            {/* Data Distribution */}
            <Card>
                <CardHeader><CardTitle className="flex items-center gap-2"><BarChart3 className="w-5 h-5" /> Data Distribution</CardTitle></CardHeader>
                <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="text-center p-4 border rounded-lg"><Database className="w-8 h-8 text-blue-500 mx-auto mb-2" /><div className="text-xl font-bold">{structuredFiles.length}</div><div className="text-sm text-gray-600">Structured Files</div></div>
                        <div className="text-center p-4 border rounded-lg"><FileText className="w-8 h-8 text-purple-500 mx-auto mb-2" /><div className="text-xl font-bold">{unstructuredFiles.length}</div><div className="text-sm text-gray-600">Unstructured Files</div></div>
                    </div>
                </CardContent>
            </Card>

            {/* Detailed File Summary */}
            <Card>
                <CardHeader>
                    <CardTitle>Detailed File Summary</CardTitle>
                    <CardDescription>Click on a file to see its quality metrics and ingestion results.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="space-y-4">
                        {selectedFiles.map((file) => (
                            <Collapsible key={file.id} open={expandedFiles.has(file.id)} onOpenChange={() => toggleFileExpansion(file.id)}>
                                <div className="border rounded-lg p-4">
                                    <CollapsibleTrigger className="flex items-center justify-between w-full text-left">
                                        <div className="flex items-center gap-3 flex-wrap">
                                            {expandedFiles.has(file.id) ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                                            <FileText className="w-4 h-4" />
                                            <span className="font-medium">{file.name}</span>
                                            <Badge variant="outline">{file.classification}</Badge>
                                            {file.ingestionStatus === "success" 
                                                ? <Badge className="bg-green-100 text-green-800">✓ Success</Badge> 
                                                : <Badge variant="destructive">✗ Failed</Badge>}
                                        </div>
                                    </CollapsibleTrigger>

                                    <CollapsibleContent className="mt-4 pt-4 border-t">
                                        <div className="space-y-4 pl-8">
                                            {/* Quality Metrics Rendering (Unchanged) */}

                                            {/* **[FIXED]** Ingestion Details Rendering Logic */}
                                            {file.ingestionStatus === "success" && (
                                                <div>
                                                    <h4 className="font-semibold mb-2">Ingestion Results</h4>
                                                    <div className="space-y-3">
                                                        {getIngestionDetailsAsArray(file.ingestionDetails).map((details, index) => (
                                                            <IngestionDetailView key={index} details={details} />
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {/* Error Message Rendering (Unchanged) */}
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
