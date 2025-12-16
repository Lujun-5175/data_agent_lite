import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { ChatInterface } from './components/ChatInterface';
import { DataUpload } from './components/DataUpload';
import { VisualizationPanel } from './components/VisualizationPanel';
import { TelcoData } from './data/mockData';
import { API_ENDPOINTS } from './config/api';

export default function App() {
  const [uploadedData, setUploadedData] = useState<TelcoData[] | null>(null);
  const [chatClearTrigger, setChatClearTrigger] = useState(0);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);

  const handleDataUpload = (data: TelcoData[]) => {
    setUploadedData(data);
  };

  const handleDataClear = () => {
    setUploadedData(null);
  };

  const handleChatClear = () => {
    setChatClearTrigger(prev => prev + 1);
    setGeneratedImage(null);
  };

  const handleImageGenerated = (imageUrl: string) => {
    setGeneratedImage(imageUrl);
  };

  return (
    <div className="h-screen w-screen overflow-hidden">
      <div style={{ transform: 'scale(0.9)', transformOrigin: 'top left', width: '111.11%', height: '111.11%' }} className="flex flex-col bg-gradient-to-br from-gray-950 via-amber-950/60 to-black relative">
        {/* Subtle glow overlay */}
        <div className="absolute inset-0 bg-gradient-to-tr from-amber-400/15 via-yellow-400/8 to-orange-400/12 pointer-events-none"></div>
        
        <Header onChatClear={handleChatClear} />
        
        <div className="flex-1 flex overflow-hidden gap-4 p-4 min-h-0 relative z-10">
          {/* Left Panel - Chat Interface */}
          <div className="w-1/2 flex flex-col min-h-0">
            <div className="h-full bg-gray-950/80 backdrop-blur-md rounded-2xl border border-amber-400/30 shadow-2xl shadow-amber-400/15 overflow-hidden">
              <ChatInterface clearTrigger={chatClearTrigger} onImageGenerated={handleImageGenerated} />
            </div>
          </div>
          
          {/* Right Panel - Data Analysis */}
          <div className="w-1/2 flex flex-col gap-4 min-h-0">
            {/* Top Half - Visualization */}
            <div className="flex-1 bg-gray-950/80 backdrop-blur-md rounded-2xl border border-amber-400/30 shadow-2xl shadow-amber-400/15 overflow-hidden">
              <VisualizationPanel uploadedData={uploadedData} generatedImage={generatedImage} />
            </div>
            
            {/* Bottom Half - Data Upload/Display */}
            <div className="flex-1 bg-gray-950/80 backdrop-blur-md rounded-2xl border border-amber-400/30 shadow-2xl shadow-amber-400/15 overflow-hidden">
              <DataUpload
                onDataUpload={handleDataUpload}
                onDataClear={handleDataClear}
                uploadedData={uploadedData}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}