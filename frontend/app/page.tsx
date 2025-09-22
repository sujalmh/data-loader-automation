"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  FolderOpen,
  FileText,
  Database,
  CheckCircle,
  Download,
  Filter,
  ArrowRight,
  ArrowLeft,
  UploadCloud, // New icon for the upload step
} from "lucide-react";
import FileSelection from "@/components/folder-selection";
import FileProcessing from "@/components/file-processing";
import FileSelectionForIngestion from "@/components/file-selection-ingestion";
import DatabaseConfiguration from "@/components/database-configuration";
import IngestionProcess from "@/components/ingestion-process";
import SummaryView from "@/components/summary-view";

// --- Type Definitions ---
export type AnalysisData = {
  file_name?: string | null;
  content_type?: string | null;
  domain?: string | null;
  subdomain?: string | null;
  intents?: string | string[] | null;
  publishing_authority?: string | null;
  published_date?: string | null;
  period_of_reference?: string | null;
  brief_summary?: string | null;
  document_size?: string | null;
  extra_fields?: Record<string, any>;
  error?: string | null;
};

export type FileData = {
  id: string;
  fileData?: string;
  name: string;
  path: string;
  size: number;
  type: string;
  selected: boolean;
  processed: boolean;
  file?: File;
  uploaded?: boolean;
  qualityMetrics?: {
    parseAccuracy: number;
    complexity: number;
  };
  classification?: "Structured" | "Unstructured";
  ingestionStatus?: "pending" | "success" | "failed";
  ingestionDetails?: IngestionDetails;
  analysis?: AnalysisData;
  error?: string;
  sourceUrl?: string;
};

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

export type IngestionDetails = (
  | StructuredIngestionDetails
  | UnstructuredIngestionDetails
) & {
  startTime: string;
  endTime: string;
};

export type FileIngestionResult = {
    fileName: string;
    fileId: string;
    fileSize: number;
    status: "success" | "failed";
    ingestionDetails: IngestionDetails | null;
    error: string | null;
    progress: number;
};


export type DatabaseConfig = {
  structured: {
    type: "postgresql" | "mysql";
    host: string;
    port: number;
    database: string;
    username: string;
    password: string;
  };
  unstructured: {
    type: "milvus" | "qdrant";
    host: string;
    port: number;
    collection: string;
    apiKey?: string;
  };
};

// --- New FileUpload Component ---
// This component handles the new "Upload Files" step.
const steps = [
  { id: 1, title: "Select & Upload", icon: FolderOpen },
  { id: 2, title: "File Processing", icon: FileText },
  { id: 3, title: "Filter Files", icon: Filter },
  { id: 4, title: "Ingestion", icon: ArrowRight },
  { id: 5, title: "Summary", icon: CheckCircle },
];

export default function DataLoaderAutomation() {
  const [currentStep, setCurrentStep] = useState(1);
  const [files, setFiles] = useState<FileData[]>([]);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [ingestionProgress, setIngestionProgress] = useState(0);
  const [uploadIsComplete, setUploadIsComplete] = useState(false);

  const nextStep = () => {
    if (currentStep < steps.length) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        // You can proceed only if files are selected and the upload is complete.
        return files.some((f) => f.selected) && uploadIsComplete;
      case 2:
        return files
          .filter((f) => f.selected && f.uploaded)
          .every((f) => f.processed);
      case 3:
        return files.some((f) => f.selected);
      case 4:
        return files
          .filter((f) => f.selected)
          .some((f) => f.ingestionStatus === "success");
      default:
        return true;
    }
  };

  useEffect(() => {
    // Automatically check if the upload requirement is met whenever files change
    const selectedFiles = files.filter((f) => f.selected);
    if (
      currentStep === 1 &&
      selectedFiles.length > 0 &&
      selectedFiles.every((f) => f.uploaded)
    ) {
      setUploadIsComplete(true);
    } else if (currentStep === 1) {
      setUploadIsComplete(false);
    }
  }, [files, currentStep]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100  p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Data Loader Automation
          </h1>
          <p className="text-lg text-gray-600">
            Automated file processing and database ingestion pipeline
          </p>
        </div>

        <div className="mb-8">
          <div className="flex items-center justify-center">
            {steps.map((step, index) => {
              const Icon = step.icon;
              const isActive = currentStep === step.id;
              const isCompleted = currentStep > step.id;
              return (
                <div
                  key={step.id}
                  className={`flex items-center ${
                    index === steps.length - 1 ? "" : "flex-1"
                  }`}
                >
                  <div className="flex flex-col items-center">
                    <div
                      className={`flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all ${
                        isActive
                          ? "bg-blue-600 border-blue-600 text-white"
                          : isCompleted
                          ? "bg-green-600 border-green-600 text-white"
                          : "bg-white border-gray-300 text-gray-400"
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                    </div>
                    <p
                      className={`text-xs mt-2 text-center ${
                        isActive
                          ? "text-blue-600 font-semibold"
                          : isCompleted
                          ? "text-green-600"
                          : "text-gray-500"
                      }`}
                    >
                      {step.title}
                    </p>
                  </div>
                  {index < steps.length - 1 && (
                    <div
                      className={`flex-1 h-0.5 mx-4 transition-all ${
                        isCompleted ? "bg-green-600" : "bg-gray-300"
                      }`}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <Card className="mb-8 min-h-[400px] flex items-center justify-center">
          <CardContent className="p-6 w-full">
            {currentStep === 1 && (
              <FileSelection
                files={files}
                setFiles={setFiles}
                onUploadComplete={() => setUploadIsComplete(true)}
              />
            )}
            {currentStep === 2 && (
              <FileProcessing
                files={files}
                setFiles={setFiles}
                progress={processingProgress}
                setProgress={setProcessingProgress}
              />
            )}
            {currentStep === 3 && (
              <FileSelectionForIngestion files={files} setFiles={setFiles} />
            )}
            {currentStep === 4 && (
              <IngestionProcess
                files={files}
                setFiles={setFiles}
                progress={ingestionProgress}
                setProgress={setIngestionProgress}
              />
            )}
            {currentStep === 5 && <SummaryView files={files} />}
          </CardContent>
        </Card>

        <div className="flex justify-between">
          <Button
            variant="outline"
            onClick={prevStep}
            disabled={currentStep === 1}
            className="flex items-center gap-2 bg-white"
          >
            <ArrowLeft className="w-4 h-4" />
            Previous
          </Button>

          {currentStep < steps.length ? (
            <Button
              onClick={nextStep}
              disabled={!canProceed()}
              className="flex items-center gap-2"
            >
              Next
              <ArrowRight className="w-4 h-4" />
            </Button>
          ) : (
            <Button variant="default" className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export Report
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
