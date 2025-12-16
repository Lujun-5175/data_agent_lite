import { useState, useRef } from 'react';
import { Button } from './ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { ScrollArea } from './ui/scroll-area';
import { Upload, Trash2 } from 'lucide-react';
import { TelcoData } from '../data/mockData';
import { API_ENDPOINTS } from '../config/api';

interface DataUploadProps {
  onDataUpload: (data: TelcoData[]) => void;
  onDataClear: () => void;
  uploadedData: TelcoData[] | null;
}

export function DataUpload({ onDataUpload, onDataClear, uploadedData }: DataUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = async (file: File) => {
    if (!file.name.endsWith('.csv')) {
      alert('请上传CSV格式的文件');
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(API_ENDPOINTS.UPLOAD, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const result = await response.json();
      console.log('Upload result:', result);

      // 直接使用上传响应中的 preview 数据
      if (result.status === 'success' && result.preview && Array.isArray(result.preview)) {
        onDataUpload(result.preview);
        if (result.message) {
          alert(result.message);
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('文件上传失败，请检查后端服务是否正常运行');
    } finally {
      setIsUploading(false);
    }
  };



  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleClearData = async () => {
    onDataClear();
  };

  if (!uploadedData) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileInputChange}
          className="hidden"
        />
        <div
          className={`w-full max-w-sm p-8 border-2 border-dashed rounded-2xl text-center transition-all duration-300 ${
            isDragOver 
              ? 'border-amber-400 bg-amber-500/20 backdrop-blur-md scale-105 shadow-lg shadow-amber-500/25' 
              : 'border-amber-500/30 hover:border-amber-400/50 bg-slate-800/40 backdrop-blur-md hover:bg-slate-800/60 shadow-lg'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="w-16 h-16 bg-gradient-to-r from-amber-500/20 to-orange-500/20 rounded-2xl flex items-center justify-center mx-auto mb-6 border border-amber-500/30 shadow-lg shadow-amber-500/10">
            <Upload className="w-8 h-8 text-amber-500" />
          </div>
          <h4 className="mb-3 text-white font-medium">上传数据集</h4>
          <p className="text-white/60 text-sm mb-6">
            支持CSV格式，拖拽或点击上传
          </p>
          <Button 
            onClick={handleFileUpload}
            disabled={isUploading}
            className="bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-700 hover:to-orange-700 rounded-xl px-6 py-2 shadow-lg shadow-amber-500/25 hover:shadow-amber-500/40 border border-amber-400/30"
          >
            {isUploading ? '上传中...' : '选择文件'}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* 数据统计信息 */}
      <div className="p-4 flex items-center justify-between border-b border-amber-400/25">
        <h3 className="text-white font-medium text-lg bg-gradient-to-r from-amber-200 to-yellow-200 bg-clip-text text-transparent">
          部分数据展示（前10行）
        </h3>
        <Button
          onClick={handleClearData}
          variant="outline"
          size="sm"
          className="text-red-400 hover:text-red-300 hover:bg-red-500/20 border-red-500/50 hover:border-red-400 bg-transparent rounded-lg h-8 w-8 p-0 shadow-lg shadow-red-500/25"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </Button>
      </div>

      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full custom-scrollbar">
          <div className="bg-slate-800/40 backdrop-blur-md rounded-xl m-3 border border-amber-500/20 overflow-hidden shadow-2xl shadow-amber-500/10">
          <Table>
            <TableHeader>
              <TableRow className="border-amber-500/20 hover:bg-slate-700/50">
                {uploadedData.length > 0 && Object.keys(uploadedData[0]).map((key) => (
                  <TableHead key={key} className="text-amber-200 text-xs">
                    {key}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {uploadedData.slice(0, 10).map((row, index) => (
                <TableRow key={index} className="border-amber-500/20 hover:bg-slate-700/30 transition-colors">
                  {Object.entries(row).map(([key, value], cellIndex) => (
                    <TableCell key={cellIndex} className="text-xs text-white/90">
                      {key === 'Churn' && (value === 'Yes' || value === 'No') ? (
                        <span className={`px-2 py-1 rounded-lg text-xs font-medium border shadow-lg ${
                          value === 'Yes' 
                            ? 'bg-red-600/30 text-red-300 border-red-500/50 shadow-red-500/25' 
                            : 'bg-green-600/30 text-green-300 border-green-500/50 shadow-green-500/25'
                        }`}>
                          {value}
                        </span>
                      ) : (
                        String(value)
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
