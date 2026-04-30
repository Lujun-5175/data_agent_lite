import { useState } from 'react';
import { Header } from './components/Header';
import { ChatInterface } from './components/ChatInterface';

export default function App() {
  const [chatClearTrigger, setChatClearTrigger] = useState(0);

  return (
    <div className="fixed inset-0 flex w-screen overflow-hidden bg-[radial-gradient(circle_at_top,#ffffff_0%,#f8fafc_32%,#eef3f9_100%)] text-slate-900">
      <div className="flex min-h-0 w-full flex-1 flex-col overflow-hidden">
        <Header onChatClear={() => setChatClearTrigger((prev) => prev + 1)} />
        <main className="h-full min-h-0 flex-1 overflow-hidden">
          <ChatInterface clearTrigger={chatClearTrigger} />
        </main>
      </div>
    </div>
  );
}
