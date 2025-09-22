"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Filter, Search, SortAsc, SortDesc } from "lucide-react"
import type { FileData } from "@/app/page"

interface FileSelectionForIngestionProps {
  files: FileData[]
  setFiles: (files: FileData[]) => void
}

export default function FileSelectionForIngestion({ files, setFiles }: FileSelectionForIngestionProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState<string>("all")
  const [sortBy, setSortBy] = useState<string>("name")
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc")

  const processedFiles = files.filter((f) => f.processed)

  const toggleFileSelection = (fileId: string) => {
    setFiles(files.map((file) => (file.id === fileId ? { ...file, selected: !file.selected } : file)))
  }

  const toggleAllFiles = (checked: boolean) => {
    setFiles(files.map((file) => (file.processed ? { ...file, selected: checked } : file)))
  }

  const getQualityScore = (file: FileData) => {
    if (!file.qualityMetrics) return 0
    return (
      (file.qualityMetrics.parseAccuracy)
    )
  }

  const filteredAndSortedFiles = processedFiles
    .filter((file) => {
      const matchesSearch = file.name.toLowerCase().includes(searchTerm.toLowerCase())
      const matchesFilter = filterType === "all"
      return matchesSearch && matchesFilter
    })
    .sort((a, b) => {
      let aValue, bValue

      switch (sortBy) {
        case "name":
          aValue = a.name
          bValue = b.name
          break
        case "quality":
          aValue = getQualityScore(a)
          bValue = getQualityScore(b)
          break
        default:
          aValue = a.name
          bValue = b.name
      }

      if (sortOrder === "asc") {
        return aValue < bValue ? -1 : aValue > bValue ? 1 : 0
      } else {
        return aValue > bValue ? -1 : aValue < bValue ? 1 : 0
      }
    })

  const selectedCount = processedFiles.filter((f) => f.selected).length
  const totalCount = processedFiles.length;
  
  const getClassificationBadge = (classification: string) => {
    switch (classification) {
      case "Structured":
        return <Badge className="bg-blue-500">Structured</Badge>
      case "Semi-Structured":
        return <Badge className="bg-purple-500">Semi-Structured</Badge>
      case "Unstructured":
        return <Badge className="bg-gray-500">Unstructured</Badge>
      default:
        return <Badge variant="outline">Unknown</Badge>
    }
  }

  const getQualityBadge = (score: number) => {
    if (score >= 2.5) return <Badge className="bg-green-500">High</Badge>
    if (score >= 1.5) return <Badge className="bg-yellow-500">Medium</Badge>
    return <Badge className="bg-red-500">Low</Badge>
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Step 3: File Selection for Ingestion</h2>
        <p className="text-gray-600">Choose which processed files to ingest into databases</p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-gray-600">{totalCount}</div>
            <div className="text-sm text-gray-600">Files</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="text-2xl font-bold text-green-600">{selectedCount}</div>
            <div className="text-sm text-gray-600">Selected for Ingestion</div>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="w-5 h-5" />
            Filter & Sort Options
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="select-all-processed"
                checked={selectedCount === processedFiles.length}
                onCheckedChange={toggleAllFiles}
              />
              <label htmlFor="select-all-processed" className="text-sm font-medium">
                Select All Processed Files
              </label>
            </div>

            <div className="flex items-center gap-2">
              <Search className="w-4 h-4 text-gray-400" />
              <Input
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-48"
              />
            </div>

            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Filter by type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="Structured">Structured</SelectItem>
                <SelectItem value="Semi-Structured">Semi-Structured</SelectItem>
                <SelectItem value="Unstructured">Unstructured</SelectItem>
              </SelectContent>
            </Select>

            <Select value={sortBy} onValueChange={setSortBy}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="name">Name</SelectItem>
                <SelectItem value="type">Classification</SelectItem>
                <SelectItem value="quality">Quality Score</SelectItem>
              </SelectContent>
            </Select>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setSortOrder(sortOrder === "asc" ? "desc" : "asc")}
              className="flex items-center gap-2"
            >
              {sortOrder === "asc" ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />}
              {sortOrder === "asc" ? "Ascending" : "Descending"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* File Selection Table */}
      <Card>
        <CardHeader>
          <CardTitle>Processed Files</CardTitle>
          <CardDescription>
            Select files for database ingestion ({filteredAndSortedFiles.length} files shown)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">Select</TableHead>
                <TableHead>File Name</TableHead>
                <TableHead>Classification</TableHead>
                <TableHead>Intents</TableHead>
                <TableHead>Brief Description</TableHead>
                <TableHead>Quality Score</TableHead>
                <TableHead>Size</TableHead>
                
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAndSortedFiles.map((file) => {
                const qualityScore = getQualityScore(file)

                return (
                  <TableRow key={file.id}>
                    <TableCell>
                      <Checkbox checked={file.selected} onCheckedChange={() => toggleFileSelection(file.id)} />
                    </TableCell>
                    <TableCell className="font-medium">{file.name}</TableCell>
                    <TableCell>
                      {file.classification && (
                          file.classification
                        )}
                    </TableCell>
                    <TableCell>                                
                      <div className="flex flex-col gap-1 ">
                        {file.analysis?.intents && (
                          <>
                            {Array.isArray(file.analysis.intents)
                              ? file.analysis.intents.map((intent, idx) => (
                                  <Badge key={idx} variant="default" className="w-40">{intent}</Badge>
                                ))
                              : <Badge variant="default">{file.analysis.intents}</Badge>}
                          </>
                        )}
                      </div>
                    </TableCell>
                    
                    <TableCell>
                      {file.analysis?.brief_summary && (
                          file.analysis?.brief_summary
                        )}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <span>{qualityScore.toFixed(1)}/3</span>
                        {getQualityBadge(qualityScore)}
                      </div>
                    </TableCell>
                    <TableCell>{(file.size / 1024 / 1024).toFixed(1)} MB</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
