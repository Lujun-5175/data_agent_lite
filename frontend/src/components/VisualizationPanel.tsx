import { useState, useEffect } from 'react';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ReferenceLine } from 'recharts';
import { TelcoData, columnInfo, mockAnalysisResults } from '../data/mockData';
import { BarChart3, TrendingUp, Activity, Image as ImageIcon, RefreshCw } from 'lucide-react';

interface VisualizationPanelProps {
  uploadedData: TelcoData[] | null;
  generatedImage?: string | null;
}

export function VisualizationPanel({ uploadedData, generatedImage }: VisualizationPanelProps) {
  const [variable1, setVariable1] = useState<string>('');
  const [variable2, setVariable2] = useState<string>('');
  const [chartData, setChartData] = useState<any[]>([]);
  const [analysisResult, setAnalysisResult] = useState<{ correlation: number; conclusion: string } | null>(null);
  const [showInteractiveChart, setShowInteractiveChart] = useState<boolean>(false);

  const allColumns = [...columnInfo.categorical, ...columnInfo.numerical];
  
  const isNumerical = (column: string) => columnInfo.numerical.includes(column);
  const isCategorical = (column: string) => columnInfo.categorical.includes(column);

  // 当有新的 AI 生成图片时，自动切换到显示 AI 图片
  useEffect(() => {
    if (generatedImage) {
      setShowInteractiveChart(false);
    }
  }, [generatedImage]);

  useEffect(() => {
    if (variable1 && variable2 && uploadedData) {
      generateChartData();
      getAnalysisResult();
    }
  }, [variable1, variable2, uploadedData]);

  const generateChartData = () => {
    if (!uploadedData || !variable1 || !variable2) return;

    if (isCategorical(variable1) && isCategorical(variable2)) {
      // Stacked bar chart data for two categorical variables
      const groupedData = uploadedData.reduce((acc, row) => {
        const key = row[variable1 as keyof TelcoData] as string;
        const value = row[variable2 as keyof TelcoData] as string;
        
        if (!acc[key]) acc[key] = {};
        if (!acc[key][value]) acc[key][value] = 0;
        acc[key][value]++;
        
        return acc;
      }, {} as Record<string, Record<string, number>>);

      const chartData = Object.keys(groupedData).map(key => {
        const item: any = { name: key };
        Object.keys(groupedData[key]).forEach(value => {
          item[value] = groupedData[key][value];
        });
        return item;
      });

      setChartData(chartData);
    } else if (isNumerical(variable1) && isNumerical(variable2)) {
      // Scatter plot data for two numerical variables
      const scatterData = uploadedData.map((row, index) => ({
        x: row[variable1 as keyof TelcoData] as number,
        y: row[variable2 as keyof TelcoData] as number,
        id: index
      }));

      setChartData(scatterData);
    } else {
      // Box plot or grouped bar chart for mixed types
      const categoricalVar = isCategorical(variable1) ? variable1 : variable2;
      const numericalVar = isNumerical(variable1) ? variable1 : variable2;
      
      const groupedData = uploadedData.reduce((acc, row) => {
        const category = row[categoricalVar as keyof TelcoData] as string;
        const value = row[numericalVar as keyof TelcoData] as number;
        
        if (!acc[category]) acc[category] = [];
        acc[category].push(value);
        
        return acc;
      }, {} as Record<string, number[]>);

      const chartData = Object.keys(groupedData).map(category => {
        const values = groupedData[category];
        return {
          name: category,
          average: values.reduce((sum, val) => sum + val, 0) / values.length,
          min: Math.min(...values),
          max: Math.max(...values),
          count: values.length
        };
      });

      setChartData(chartData);
    }
  };

  const getAnalysisResult = () => {
    const key1 = `${variable1}_${variable2}`;
    const key2 = `${variable2}_${variable1}`;
    
    const result = mockAnalysisResults[key1 as keyof typeof mockAnalysisResults] || 
                   mockAnalysisResults[key2 as keyof typeof mockAnalysisResults];
    
    if (result) {
      setAnalysisResult(result);
    } else {
      // Generate a mock result
      const correlation = Math.random() * 2 - 1; // Random correlation between -1 and 1
      setAnalysisResult({
        correlation: Number(correlation.toFixed(2)),
        conclusion: `${variable1}与${variable2}之间的相关性为${Math.abs(correlation) > 0.5 ? '强' : Math.abs(correlation) > 0.3 ? '中等' : '弱'}`
      });
    }
  };

  const renderChart = () => {
    if (!chartData.length || !variable1 || !variable2) return null;

    if (isCategorical(variable1) && isCategorical(variable2)) {
      const uniqueValues = Array.from(new Set(
        uploadedData?.map(row => row[variable2 as keyof TelcoData] as string) || []
      ));
      
      const colors = ['#F59E0B', '#EAB308', '#F97316', '#84CC16', '#06B6D4'];

      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#ffffff20" />
            <XAxis dataKey="name" stroke="#ffffff80" fontSize={11} />
            <YAxis stroke="#ffffff80" fontSize={11} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                border: '1px solid rgba(245, 158, 11, 0.3)',
                borderRadius: '12px',
                color: '#ffffff',
                backdropFilter: 'blur(12px)'
              }} 
            />
            {uniqueValues.map((value, index) => (
              <Bar 
                key={value} 
                dataKey={value} 
                stackId="a" 
                fill={colors[index % colors.length]} 
                radius={[2, 2, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      );
    } else if (isNumerical(variable1) && isNumerical(variable2)) {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#ffffff20" />
            <XAxis dataKey="x" name={variable1} stroke="#ffffff80" fontSize={11} />
            <YAxis dataKey="y" name={variable2} stroke="#ffffff80" fontSize={11} />
            <Tooltip 
              cursor={{ strokeDasharray: '2 2' }}
              contentStyle={{ 
                backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                border: '1px solid rgba(245, 158, 11, 0.3)',
                borderRadius: '12px',
                color: '#ffffff',
                backdropFilter: 'blur(12px)'
              }} 
            />
            <Scatter fill="#F59E0B" />
            <ReferenceLine stroke="#EAB308" strokeDasharray="3 3" />
          </ScatterChart>
        </ResponsiveContainer>
      );
    } else {
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="2 2" stroke="#ffffff20" />
            <XAxis dataKey="name" stroke="#ffffff80" fontSize={11} />
            <YAxis stroke="#ffffff80" fontSize={11} />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(30, 41, 59, 0.9)', 
                border: '1px solid rgba(245, 158, 11, 0.3)',
                borderRadius: '12px',
                color: '#ffffff',
                backdropFilter: 'blur(12px)'
              }} 
            />
            <Bar dataKey="average" fill="#F59E0B" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );
    }
  };

  const getChartTitle = () => {
    if (!variable1 || !variable2) return '请选择两个变量进行分析';
    
    if (isCategorical(variable1) && isCategorical(variable2)) {
      return `${variable1} vs ${variable2} - 堆叠柱状图`;
    } else if (isNumerical(variable1) && isNumerical(variable2)) {
      return `${variable1} vs ${variable2} - 散点图`;
    } else {
      return `${variable1} vs ${variable2} - 分组分析`;
    }
  };

  const getChartIcon = () => {
    if (!variable1 || !variable2) {
      return <BarChart3 className="w-3 h-3 text-white" />;
    }
    
    if (isCategorical(variable1) && isCategorical(variable2)) {
      return <BarChart3 className="w-3 h-3 text-white" />;
    } else if (isNumerical(variable1) && isNumerical(variable2)) {
      return <TrendingUp className="w-3 h-3 text-white" />;
    } else {
      return <Activity className="w-3 h-3 text-white" />;
    }
  };

  if (!uploadedData) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-r from-amber-500/20 to-yellow-500/20 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-amber-500/30 shadow-lg shadow-amber-500/10">
            <BarChart3 className="w-8 h-8 text-amber-500" />
          </div>
          <p className="text-white/80 mb-1">请先上传数据集</p>
          <p className="text-white/50 text-sm">上传后可进行变量分析</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* 变量选择区域和视图切换 */}
      <div className="p-4 border-b border-amber-400/25 flex-shrink-0">
        <div className="flex gap-3 items-center">
          <div className="flex-1 grid grid-cols-2 gap-3">
            <Select value={variable1} onValueChange={setVariable1}>
              <SelectTrigger className="bg-gray-900/70 backdrop-blur-md border-amber-400/35 text-white rounded-xl h-9 text-sm shadow-lg">
                <SelectValue placeholder="选择变量 1" />
              </SelectTrigger>
              <SelectContent className="bg-gray-950/90 border-amber-400/35 backdrop-blur-md">
                {allColumns.map(column => (
                  <SelectItem key={column} value={column} disabled={column === variable2} className="text-white focus:bg-gray-800 text-sm">
                    {column}
                    <Badge variant="outline" className="ml-2 text-xs border-amber-300/60 text-amber-300">
                      {isNumerical(column) ? '数值' : '分类'}
                    </Badge>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={variable2} onValueChange={setVariable2}>
              <SelectTrigger className="bg-gray-900/70 backdrop-blur-md border-amber-400/35 text-white rounded-xl h-9 text-sm shadow-lg">
                <SelectValue placeholder="选择变量 2" />
              </SelectTrigger>
              <SelectContent className="bg-gray-950/90 border-amber-400/35 backdrop-blur-md">
                {allColumns.map(column => (
                  <SelectItem key={column} value={column} disabled={column === variable1} className="text-white focus:bg-gray-800 text-sm">
                    {column}
                    <Badge variant="outline" className="ml-2 text-xs border-amber-300/60 text-amber-300">
                      {isNumerical(column) ? '数值' : '分类'}
                    </Badge>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          {/* 视图切换按钮 - 仅在有生成图片时显示 */}
          {generatedImage && (
            <Button
              onClick={() => setShowInteractiveChart(!showInteractiveChart)}
              className="bg-gray-900/70 backdrop-blur-md border border-amber-400/35 text-white hover:bg-gray-800/70 rounded-xl h-9 px-3 shadow-lg"
              variant="outline"
            >
              {showInteractiveChart ? (
                <>
                  <ImageIcon className="w-4 h-4 mr-1" />
                  <span className="text-xs">AI图表</span>
                </>
              ) : (
                <>
                  <BarChart3 className="w-4 h-4 mr-1" />
                  <span className="text-xs">交互图表</span>
                </>
              )}
            </Button>
          )}
        </div>
      </div>

      {/* 图表和分析结果区域 */}
      <div className="flex-1 p-4 overflow-hidden">
        <div className="h-full custom-scrollbar overflow-auto">
        {/* 显示 AI 生成的图片 */}
        {generatedImage && !showInteractiveChart ? (
          <div className="h-full flex items-center justify-center bg-gray-900/50 backdrop-blur-md rounded-xl border border-amber-400/25 p-4 shadow-2xl shadow-amber-400/15">
            <div className="w-full h-full flex flex-col">
              <div className="flex items-center gap-2 mb-3 px-2">
                <div className="w-6 h-6 bg-gradient-to-r from-amber-400 to-yellow-400 rounded-lg flex items-center justify-center shadow-lg shadow-amber-400/30">
                  <ImageIcon className="w-3 h-3 text-white" />
                </div>
                <span className="text-white text-sm font-medium">AI 生成图表</span>
              </div>
              <div className="flex-1 flex items-center justify-center">
                <img 
                  src={generatedImage} 
                  alt="AI Generated Chart" 
                  className="max-w-full max-h-full object-contain rounded-lg shadow-lg"
                  onError={(e) => {
                    console.error('Image load error:', generatedImage);
                    e.currentTarget.style.display = 'none';
                    e.currentTarget.parentElement?.insertAdjacentHTML('afterbegin', 
                      '<div class="text-white/60 text-center"><p class="mb-2">图片加载失败</p><p class="text-sm text-white/40">URL: ' + generatedImage + '</p></div>'
                    );
                  }}
                />
              </div>
            </div>
          </div>
        ) : variable1 && variable2 ? (
          <div className="h-full flex gap-4">
            {/* 左侧图表区域 */}
            <div className="flex-1 bg-gray-900/50 backdrop-blur-md rounded-xl border border-amber-400/25 p-4 shadow-2xl shadow-amber-400/15 overflow-hidden">
              {renderChart()}
            </div>
            
            {/* 右侧分析结果区域 */}
            {analysisResult && (
              <div className="w-64 bg-gray-900/70 backdrop-blur-md rounded-xl border border-amber-400/25 p-4 shadow-lg shadow-amber-400/15 overflow-hidden">
                <div className="flex flex-col h-full">
                  <div className="custom-scrollbar overflow-y-auto flex-1">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-6 h-6 bg-gradient-to-r from-amber-400 to-yellow-400 rounded-lg flex items-center justify-center shadow-lg shadow-amber-400/30">
                        {getChartIcon()}
                      </div>
                      <span className="text-white text-sm font-medium">分析结果</span>
                    </div>
                    
                    <div className="space-y-4">
                      <div className="bg-slate-700/50 rounded-lg p-3 border border-amber-500/20">
                        <div className="text-xs text-amber-300 mb-1">相关性系数</div>
                        <div className={`text-lg font-bold ${Math.abs(analysisResult.correlation) > 0.5 ? 'text-red-400' : Math.abs(analysisResult.correlation) > 0.3 ? 'text-yellow-400' : 'text-green-400'}`}>
                          {analysisResult.correlation}
                        </div>
                      </div>
                      
                      <div className="bg-slate-700/50 rounded-lg p-3 border border-amber-500/20">
                        <div className="text-xs text-amber-300 mb-1">分析结论</div>
                        <div className="text-sm text-white leading-relaxed">{analysisResult.conclusion}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-amber-500/20 to-yellow-500/20 rounded-2xl flex items-center justify-center mx-auto mb-4 border border-amber-500/30 shadow-lg shadow-amber-500/10">
                <BarChart3 className="w-8 h-8 text-amber-500" />
              </div>
              <p className="text-white/60">请选择两个变量开始分析</p>
            </div>
          </div>
        )}
        </div>
      </div>
    </div>
  );
}