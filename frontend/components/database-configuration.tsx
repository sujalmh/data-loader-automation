"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Database, TestTube } from "lucide-react"
import type { DatabaseConfig } from "@/app/page"
import { useState } from "react"

interface DatabaseConfigurationProps {
  config: DatabaseConfig
  setConfig: (config: DatabaseConfig) => void
}

export default function DatabaseConfiguration({ config, setConfig }: DatabaseConfigurationProps) {
  const [structuredTestStatus, setStructuredTestStatus] = useState<"idle" | "testing" | "success" | "failed">("idle")
  const [unstructuredTestStatus, setUnstructuredTestStatus] = useState<"idle" | "testing" | "success" | "failed">(
    "idle",
  )

  const updateStructuredConfig = (field: string, value: string | number) => {
    setConfig({
      ...config,
      structured: {
        ...config.structured,
        [field]: value,
      },
    })
  }

  const updateUnstructuredConfig = (field: string, value: string | number) => {
    setConfig({
      ...config,
      unstructured: {
        ...config.unstructured,
        [field]: value,
      },
    })
  }

  const testStructuredConnection = async () => {
    setStructuredTestStatus("testing")
    // Simulate connection test
    await new Promise((resolve) => setTimeout(resolve, 2000))
    setStructuredTestStatus(Math.random() > 0.3 ? "success" : "failed")
  }

  const testUnstructuredConnection = async () => {
    setUnstructuredTestStatus("testing")
    // Simulate connection test
    await new Promise((resolve) => setTimeout(resolve, 2000))
    setUnstructuredTestStatus(Math.random() > 0.3 ? "success" : "failed")
  }

  const getConnectionStatus = (status: string) => {
    switch (status) {
      case "testing":
        return <Badge className="bg-yellow-500">Testing...</Badge>
      case "success":
        return <Badge className="bg-green-500">✓ Connected</Badge>
      case "failed":
        return <Badge variant="destructive">✗ Failed</Badge>
      default:
        return <Badge variant="outline">Not Tested</Badge>
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold mb-2">Step 4: Database Connection Configuration</h2>
        <p className="text-gray-600">Configure connections for structured and unstructured data storage</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Structured Database Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-500" />
              Structured Database
            </CardTitle>
            <CardDescription>Configuration for relational database (tables, schemas)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="structured-type">Database Type</Label>
              <Select value={config.structured.type} onValueChange={(value) => updateStructuredConfig("type", value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="postgresql">PostgreSQL</SelectItem>
                  <SelectItem value="mysql">MySQL</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="structured-host">Host</Label>
                <Input
                  id="structured-host"
                  value={config.structured.host}
                  onChange={(e) => updateStructuredConfig("host", e.target.value)}
                  placeholder="localhost"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="structured-port">Port</Label>
                <Input
                  id="structured-port"
                  type="number"
                  value={config.structured.port}
                  onChange={(e) => updateStructuredConfig("port", Number.parseInt(e.target.value))}
                  placeholder="5432"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="structured-database">Database Name</Label>
              <Input
                id="structured-database"
                value={config.structured.database}
                onChange={(e) => updateStructuredConfig("database", e.target.value)}
                placeholder="dataloader"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="structured-username">Username</Label>
                <Input
                  id="structured-username"
                  value={config.structured.username}
                  onChange={(e) => updateStructuredConfig("username", e.target.value)}
                  placeholder="postgres"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="structured-password">Password</Label>
                <Input
                  id="structured-password"
                  type="password"
                  value={config.structured.password}
                  onChange={(e) => updateStructuredConfig("password", e.target.value)}
                  placeholder="••••••••"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-4">
              <Button
                variant="outline"
                onClick={testStructuredConnection}
                disabled={structuredTestStatus === "testing"}
                className="flex items-center gap-2 bg-transparent"
              >
                <TestTube className="w-4 h-4" />
                Test Connection
              </Button>
              {getConnectionStatus(structuredTestStatus)}
            </div>
          </CardContent>
        </Card>

        {/* Unstructured Database Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="w-5 h-5 text-purple-500" />
              Unstructured Database
            </CardTitle>
            <CardDescription>Configuration for vector database (embeddings, documents)</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="unstructured-type">Database Type</Label>
              <Select
                value={config.unstructured.type}
                onValueChange={(value) => updateUnstructuredConfig("type", value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="milvus">Milvus</SelectItem>
                  <SelectItem value="qdrant">Qdrant</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="unstructured-host">Host</Label>
                <Input
                  id="unstructured-host"
                  value={config.unstructured.host}
                  onChange={(e) => updateUnstructuredConfig("host", e.target.value)}
                  placeholder="localhost"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="unstructured-port">Port</Label>
                <Input
                  id="unstructured-port"
                  type="number"
                  value={config.unstructured.port}
                  onChange={(e) => updateUnstructuredConfig("port", Number.parseInt(e.target.value))}
                  placeholder="19530"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="unstructured-collection">Collection Name</Label>
              <Input
                id="unstructured-collection"
                value={config.unstructured.collection}
                onChange={(e) => updateUnstructuredConfig("collection", e.target.value)}
                placeholder="documents"
              />
            </div>

            {config.unstructured.type === "qdrant" && (
              <div className="space-y-2">
                <Label htmlFor="unstructured-apikey">API Key (Optional)</Label>
                <Input
                  id="unstructured-apikey"
                  type="password"
                  value={config.unstructured.apiKey || ""}
                  onChange={(e) => updateUnstructuredConfig("apiKey", e.target.value)}
                  placeholder="••••••••"
                />
              </div>
            )}

            <div className="flex items-center justify-between pt-4">
              <Button
                variant="outline"
                onClick={testUnstructuredConnection}
                disabled={unstructuredTestStatus === "testing"}
                className="flex items-center gap-2 bg-transparent"
              >
                <TestTube className="w-4 h-4" />
                Test Connection
              </Button>
              {getConnectionStatus(unstructuredTestStatus)}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Connection Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Connection Summary</CardTitle>
          <CardDescription>Review your database configurations before proceeding to ingestion</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 border rounded-lg">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Database className="w-4 h-4 text-blue-500" />
                Structured Database
              </h4>
              <div className="text-sm space-y-1">
                <p>
                  <strong>Type:</strong> {config.structured.type.toUpperCase()}
                </p>
                <p>
                  <strong>Host:</strong> {config.structured.host}:{config.structured.port}
                </p>
                <p>
                  <strong>Database:</strong> {config.structured.database}
                </p>
                <p>
                  <strong>Status:</strong> {getConnectionStatus(structuredTestStatus)}
                </p>
              </div>
            </div>

            <div className="p-4 border rounded-lg">
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Database className="w-4 h-4 text-purple-500" />
                Unstructured Database
              </h4>
              <div className="text-sm space-y-1">
                <p>
                  <strong>Type:</strong> {config.unstructured.type.toUpperCase()}
                </p>
                <p>
                  <strong>Host:</strong> {config.unstructured.host}:{config.unstructured.port}
                </p>
                <p>
                  <strong>Collection:</strong> {config.unstructured.collection}
                </p>
                <p>
                  <strong>Status:</strong> {getConnectionStatus(unstructuredTestStatus)}
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
