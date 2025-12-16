import { useState, useEffect, useRef } from 'react';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { ScrollArea } from './ui/scroll-area';
import { Send, Bot, User } from 'lucide-react';
import { API_ENDPOINTS } from '../config/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const initialMessages: Message[] = [
  {
    id: '1',
    type: 'assistant',
    content: '您好！我是数据分析助手，可以帮您分析上传的数据集。请在右侧上传CSV文件，然后选择变量进行分析，我会为您提供详细的分析结果和建议。',
    timestamp: new Date()
  }
];

interface ChatInterfaceProps {
  clearTrigger: number;
  onImageGenerated?: (imageUrl: string) => void;
}

export function ChatInterface({ clearTrigger, onImageGenerated }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // 监听清除触发器，重置对话
  useEffect(() => {
    if (clearTrigger > 0) {
      setMessages(initialMessages);
      setInputValue('');
      setIsLoading(false);
    }
  }, [clearTrigger]);

  // 构建消息历史（LangServe格式）
  const buildMessageHistory = () => {
    return messages
      .filter(msg => 
        msg.id !== '1' && // 排除初始欢迎消息
        msg.content.trim() !== '' // 排除空消息
      )
      .map(msg => ({
        type: msg.type === 'user' ? 'human' : 'ai',
        content: msg.content
      }));
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // 创建一个临时的 assistant 消息来显示流式内容
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      type: 'assistant',
      content: '',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, assistantMessage]);

    try {
      // 构建请求体（LangServe格式）
      const messageHistory = buildMessageHistory();
      const requestBody = {
        input: {
          messages: [
            ...messageHistory,
            { type: 'human', content: userMessage.content }
          ]
        }
      };

      console.log('🚀 发送请求:', JSON.stringify(requestBody, null, 2));

      const response = await fetch(API_ENDPOINTS.AGENT_STREAM, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      console.log('📡 响应状态:', response.status, response.statusText);
      console.log('📋 响应头:', {
        contentType: response.headers.get('content-type'),
        transferEncoding: response.headers.get('transfer-encoding'),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      console.log('📖 Reader:', reader ? '已创建' : 'undefined');
      
      const decoder = new TextDecoder();
      let accumulatedContent = '';
      let buffer = '';
      let chunkCount = 0;

      if (reader) {
        console.log('🔄 开始读取流...');
        while (true) {
          console.log(`📦 读取第 ${chunkCount + 1} 个chunk...`);
          const { done, value } = await reader.read();
          console.log(`✅ 收到chunk ${chunkCount + 1}:`, { done, valueLength: value?.length });
          
          if (done) {
            console.log('🏁 流读取完成');
            break;
          }
          
          chunkCount++;

          // 将新数据添加到缓冲区
          const decodedChunk = decoder.decode(value, { stream: true });
          console.log(`📝 解码后的原始文本长度:`, decodedChunk.length);
          console.log(`📝 前300字符:`, decodedChunk.substring(0, 300));
          console.log(`📝 是否以\\n\\n结尾:`, decodedChunk.endsWith('\n\n'));
          buffer += decodedChunk;
          
          // SSE 格式：块之间用双换行分隔
          const blocks = buffer.split('\n\n');
          console.log(`📚 分割出 ${blocks.length} 个块，buffer长度: ${buffer.length}`);
          
          // 只有当buffer不以\n\n结尾时，最后一个块才可能不完整
          // 否则所有块都是完整的（最后一个是空字符串）
          if (!buffer.endsWith('\n\n')) {
            buffer = blocks.pop() || '';
            console.log(`⏳ 保留不完整的块到buffer，长度: ${buffer.length}`);
          } else {
            buffer = '';
            console.log(`✅ 所有块都完整`);
          }
          
          console.log(`🔢 准备处理 ${blocks.length} 个块`);

          for (const block of blocks) {
            if (!block.trim()) {
              console.log('⏭️  跳过空块');
              continue;
            }
            console.log(`🔍 处理块:`, block.substring(0, 100));

            const lines = block.split('\n');
            let eventType = '';
            let dataContent = '';

            // 解析每一块中的 event 和 data
            for (const line of lines) {
              if (line.startsWith('event:')) {
                eventType = line.slice(6).trim();
              } else if (line.startsWith('data:')) {
                dataContent = line.slice(5).trim();
              }
            }

            console.log('SSE块 - 事件类型:', eventType);
            console.log('SSE块 - 原始数据:', dataContent);

            // 只处理 event: data 类型的消息
            if (eventType === 'data' && dataContent) {
              try {
                const parsed = JSON.parse(dataContent);
                console.log('✅ 解析到的完整JSON:', parsed);
                console.log('📦 JSON结构的keys:', Object.keys(parsed));
                
                // 尝试从不同可能的路径提取 messages
                // LangServe SSE 流的结构是 { model: { messages: [...] } }
                let messagesArray = parsed.model?.messages || parsed.messages || parsed.output?.messages;
                
                console.log('📨 提取到的messages数组:', messagesArray);
                
                // 从 messages 数组中提取最后一个消息
                if (messagesArray && Array.isArray(messagesArray) && messagesArray.length > 0) {
                  const lastMessage = messagesArray[messagesArray.length - 1];
                  console.log('🔍 最后一条消息:', lastMessage);
                  console.log('🔍 消息类型:', lastMessage.type);
                  console.log('🔍 消息内容:', lastMessage.content);
                  
                  // 如果有 content，显示内容
                  if (lastMessage.content && typeof lastMessage.content === 'string' && lastMessage.content.trim()) {
                    accumulatedContent = lastMessage.content;
                    console.log('更新内容为:', accumulatedContent);
                    
                    // 检测图片生成标记
                    const imageMatch = accumulatedContent.match(/IMAGE_GENERATED:\s*(\S+)/);
                    if (imageMatch && onImageGenerated) {
                      const filename = imageMatch[1];
                      const imageUrl = `http://localhost:8002/static/images/${filename}`;
                      console.log('检测到生成的图片:', imageUrl);
                      onImageGenerated(imageUrl);
                    }
                    
                    // 更新消息内容
                    setMessages(prev => prev.map(msg => 
                      msg.id === assistantMessageId 
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    ));
                  }
                  // 如果有 tool_calls，显示"分析中..."
                  else if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
                    console.log('检测到工具调用，显示分析中...');
                    if (!accumulatedContent) {
                      setMessages(prev => prev.map(msg => 
                        msg.id === assistantMessageId 
                          ? { ...msg, content: '正在分析中...' }
                          : msg
                      ));
                    }
                  }
                  // 如果是工具消息，可以显示工具执行结果（可选）
                  else if (lastMessage.type === 'tool') {
                    console.log('工具执行完毕，结果:', lastMessage.content);
                  }
                }
              } catch (e) {
                console.error('Error parsing SSE data:', e, 'Raw data:', dataContent);
              }
            } else if (eventType === 'end') {
              console.log('流结束');
              break;
            }
          }
        }
        
        // 🔥 关键修复：流结束后处理剩余的buffer
        console.log(`🔥 流结束，处理剩余buffer，长度: ${buffer.length}`);
        if (buffer.trim()) {
          console.log(`🔥 剩余buffer内容（前500字符）:`, buffer.substring(0, 500));
          
          // 直接在整个buffer中查找所有的 event: data 块
          const dataEventRegex = /event:\s*data\s*\n\s*data:\s*({[\s\S]*?})(?=\s*\n\s*event:|\s*$)/g;
          let match;
          let foundData = false;
          
          while ((match = dataEventRegex.exec(buffer)) !== null) {
            const dataContent = match[1];
            console.log('🔥 找到data事件，数据:', dataContent.substring(0, 200));
            foundData = true;
            
            try {
              const parsed = JSON.parse(dataContent);
              console.log('🔥 解析成功，keys:', Object.keys(parsed));
              const messagesArray = parsed.model?.messages || parsed.messages || parsed.output?.messages;
              
              if (messagesArray && Array.isArray(messagesArray) && messagesArray.length > 0) {
                const lastMessage = messagesArray[messagesArray.length - 1];
                console.log('🔥 最后一条消息 type:', lastMessage.type);
                console.log('🔥 最后一条消息 content:', lastMessage.content?.substring(0, 100));
                
                if (lastMessage.type === 'ai' && lastMessage.content && typeof lastMessage.content === 'string' && lastMessage.content.trim()) {
                  accumulatedContent = lastMessage.content;
                  console.log('✅✅✅ 找到AI回复，更新界面！');
                  
                  const imageMatch = accumulatedContent.match(/IMAGE_GENERATED:\s*(\S+)/);
                  if (imageMatch && onImageGenerated) {
                    const filename = imageMatch[1];
                    const imageUrl = `http://localhost:8002/static/images/${filename}`;
                    console.log('检测到生成的图片:', imageUrl);
                    onImageGenerated(imageUrl);
                  }
                  
                  setMessages(prev => prev.map(msg => 
                    msg.id === assistantMessageId 
                      ? { ...msg, content: accumulatedContent }
                      : msg
                  ));
                }
              }
            } catch (e) {
              console.error('🔥 解析出错:', e);
            }
          }
          
          if (!foundData) {
            console.error('❌ 未找到任何 event: data 块');
          }
        }
        
        console.log(`✅ 总共读取了 ${chunkCount} 个chunks`);
      } else {
        console.error('❌ 无法创建 reader - response.body 为空');
      }

      setIsLoading(false);
    } catch (error) {
      console.error('Error calling agent:', error);
      
      // 显示错误消息
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId 
          ? { ...msg, content: '抱歉，调用AI助手时出现错误。请检查后端服务是否正常运行。' }
          : msg
      ));
      
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages Area */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full p-6 custom-scrollbar">
        <div className="space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${
                message.type === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.type === 'assistant' && (
                <div className="w-8 h-8 bg-gradient-to-r from-amber-500/60 to-yellow-500/60 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg shadow-amber-400/20 border border-amber-400/30">
                  <Bot className="w-4 h-4 text-amber-100" />
                </div>
              )}
              
              <div
                className={`max-w-[80%] p-4 rounded-2xl whitespace-pre-wrap shadow-lg ${
                  message.type === 'user'
                    ? 'bg-gradient-to-r from-amber-600/80 to-orange-600/80 text-amber-50 ml-auto shadow-amber-400/20 border border-amber-400/30'
                    : 'bg-gray-900/85 backdrop-blur-md text-white border border-amber-400/25 shadow-amber-400/15'
                }`}
              >
                {message.content}
              </div>
              
              {message.type === 'user' && (
                <div className="w-8 h-8 bg-gradient-to-r from-amber-500/60 to-orange-500/60 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg shadow-amber-400/20 border border-amber-400/30">
                  <User className="w-4 h-4 text-amber-100" />
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 bg-gradient-to-r from-amber-500/60 to-yellow-500/60 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg shadow-amber-400/20 border border-amber-400/30">
                <Bot className="w-4 h-4 text-amber-100" />
              </div>
              <div className="bg-gray-900/85 backdrop-blur-md text-white p-4 rounded-2xl shadow-lg border border-amber-400/25">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-amber-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-yellow-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-2 h-2 bg-orange-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                </div>
              </div>
            </div>
          )}
        </div>
        </ScrollArea>
      </div>

      {/* Input Area */}
      <div className="p-6 border-t border-amber-400/25">
        <div className="flex gap-3">
          <Textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="请输入您的问题..."
            className="flex-1 min-h-[50px] max-h-[120px] resize-none bg-gray-900/70 backdrop-blur-md border-amber-400/35 text-white placeholder:text-white/50 rounded-xl focus:ring-2 focus:ring-amber-300/60 focus:border-amber-300/60 shadow-lg custom-scrollbar"
            disabled={isLoading}
          />
          <Button
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isLoading}
            className="self-end bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 rounded-xl px-4 shadow-lg shadow-amber-400/30 hover:shadow-amber-400/45 border border-amber-300/40"
          >
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
