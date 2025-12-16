import { Button } from './ui/button';

interface HeaderProps {
  onChatClear: () => void;
}

export function Header({ onChatClear }: HeaderProps) {
  return (
    <header className="flex items-center justify-between p-4 bg-gray-950/85 backdrop-blur-md border-b border-amber-400/25 relative z-20">
      <div className="flex items-center">
        <Button 
          onClick={onChatClear}
          variant="outline"
          className="bg-gray-800/70 hover:bg-gray-700/70 text-white border-amber-400/40 hover:border-amber-300/60 px-4 py-2 rounded-lg shadow-lg shadow-amber-400/20 hover:shadow-amber-400/30 transition-all duration-200 backdrop-blur-md"
        >
          清除对话
        </Button>
      </div>
      
      <div className="absolute left-1/2 transform -translate-x-1/2 flex items-center gap-3">
        <h1 className="text-2xl font-bold text-white">
          Data Agent
        </h1>
        <span className="text-white text-sm font-normal">
          by 烨笙
        </span>
      </div>
      
      <div className="flex items-center">
        <a 
          href="https://gitee.com/ye_sheng0839/data-agent" 
          target="_blank" 
          rel="noopener noreferrer"
          className="flex items-center justify-center w-10 h-10 hover:opacity-80 transition-opacity"
        >
          <img 
            src="/gitee.svg" 
            alt="Gitee" 
            className="w-full h-full"
          />
        </a>
      </div>
    </header>
  );
}