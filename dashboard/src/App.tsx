import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ReferenceArea, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { Activity, AlertTriangle, ShieldCheck, Send } from 'lucide-react';
import { MOCK_HEATMAP, MOCK_TRENDS, MOCK_REPORT } from './mockData';

const riskColors = {
  CRITICAL: 'text-critical bg-critical/20 border-critical',
  HIGH: 'text-high bg-high/20 border-high',
  MEDIUM: 'text-medium bg-medium/20 border-medium',
  HEALTHY: 'text-healthy bg-healthy/20 border-healthy'
};

const bgColors = {
  CRITICAL: 'bg-critical',
  HIGH: 'bg-high',
  MEDIUM: 'bg-medium',
  HEALTHY: 'bg-healthy'
};

export default function App() {
  const [inverters] = useState(MOCK_HEATMAP);
  const [selectedInv, setSelectedInv] = useState(inverters[0]);
  const [filterTier, setFilterTier] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([{ role: 'assistant', text: 'Hello! I am the SolarMind AI agent. How can I help you today?' }]);

  // Simulate WebSocket
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream/PLANT_1');
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'update') {
          // just mock a UI tick for the demo validation
          console.log("WebSocket update received:", data);
        }
      } catch (err) { }
    };
    return () => ws.close();
  }, []);

  const handleChat = (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    setChatHistory([...chatHistory, { role: 'user', text: chatInput }, { role: 'assistant', text: 'Simulated RAG answer for: ' + chatInput }]);
    setChatInput('');
  };

  const filteredInverters = filterTier ? inverters.filter(i => i.risk_level === filterTier) : inverters;
  const counts = inverters.reduce((acc, inv) => {
    acc[inv.risk_level] = (acc[inv.risk_level] || 0) + 1;
    return acc;
  }, {} as any);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col p-4 gap-4 font-sans">
      {/* Panel 1: Overview Bar */}
      <div className="flex gap-4 h-24">
        {['CRITICAL', 'HIGH', 'MEDIUM', 'HEALTHY'].map(tier => (
          <div
            key={tier}
            onClick={() => setFilterTier(filterTier === tier ? null : tier)}
            className={`flex-1 border rounded-lg p-4 cursor-pointer flex items-center justify-between transition-colors
              ${filterTier === tier ? 'ring-2 ring-white/50' : ''} ${riskColors[tier as keyof typeof riskColors]}`}
          >
            <div>
              <div className="text-sm font-bold opacity-80">{tier}</div>
              <div className="text-3xl font-black">{counts[tier] || 0}</div>
            </div>
            {tier === 'CRITICAL' && <AlertTriangle size={32} className="opacity-80" />}
            {tier === 'HEALTHY' && <ShieldCheck size={32} className="opacity-80" />}
          </div>
        ))}
      </div>

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Panel 2: Heatmap (~40%) */}
        <div className="w-2/5 border border-gray-800 rounded-lg p-4 bg-gray-900/50 flex flex-col gap-2 overflow-y-auto">
          <h2 className="text-lg font-bold mb-2">Plant Heatmap (Blocks)</h2>
          <div className="grid grid-cols-5 gap-2">
            {filteredInverters.map(inv => (
              <div
                key={inv.id}
                onClick={() => setSelectedInv(inv)}
                className={`aspect-square rounded flex items-center justify-center font-mono text-xs cursor-pointer border-2 transition-all
                  ${bgColors[inv.risk_level as keyof typeof bgColors]} 
                  ${selectedInv.id === inv.id ? 'border-white scale-110 shadow-lg' : 'border-transparent opacity-80 hover:opacity-100'}
                  ${inv.risk_level === 'CRITICAL' ? 'animate-pulse' : ''}
                `}
              >
                {inv.id.replace('INV_', '')}
              </div>
            ))}
          </div>
        </div>

        <div className="w-3/5 flex flex-col gap-4">
          {/* Panel 3: Trend Charts */}
          <div className="flex-1 border border-gray-800 rounded-lg p-4 bg-gray-900/50 flex flex-col">
            <h2 className="text-lg font-bold mb-2">48H Trends: {selectedInv.id}</h2>
            <div className="flex-1 flex flex-col gap-2 min-h-0">
              <div className="flex-1 min-h-0 relative">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={MOCK_TRENDS}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="time" hide />
                    <YAxis domain={['auto', 'auto']} fontSize={10} stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                    <ReferenceLine y={65} stroke="#ef4444" strokeDasharray="3 3" />
                    <ReferenceArea x1="T-8h" x2="T-0h" fill="#ef4444" fillOpacity={0.1} />
                    <Line type="monotone" dataKey="temperature" stroke="#f97316" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="flex-1 min-h-0 relative">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={MOCK_TRENDS}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="time" hide />
                    <YAxis domain={[0.6, 1.0]} fontSize={10} stroke="#9ca3af" />
                    <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none' }} />
                    <ReferenceLine y={0.88} stroke="#eab308" strokeDasharray="3 3" />
                    <ReferenceArea x1="T-6h" x2="T-0h" fill="#ef4444" fillOpacity={0.1} />
                    <Line type="monotone" dataKey="efficiency" stroke="#22c55e" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Panel 4: AI Diagnosis Card */}
          <div className="h-64 border border-gray-800 rounded-lg p-4 bg-gray-900/50 flex gap-4">
            <div className="flex-1 flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <h2 className="text-lg font-bold flex items-center gap-2">
                  <Activity size={18} className="text-blue-400" />
                  AI Diagnosis
                </h2>
                <div className="flex gap-2 text-xs">
                  <span className="bg-blue-900 text-blue-200 px-2 py-1 rounded">Conf: {MOCK_REPORT.confidence}</span>
                  <span className="bg-purple-900 text-purple-200 px-2 py-1 rounded">Data: {MOCK_REPORT.data_quality}</span>
                </div>
              </div>
              <p className="text-sm text-gray-300">- {MOCK_REPORT.summary}</p>
              <p className="text-sm font-semibold text-gray-200 mt-2">Root Cause: {MOCK_REPORT.root_cause}</p>
              <button className="mt-auto w-full bg-blue-600 hover:bg-blue-500 text-white py-2 rounded font-bold transition-colors">
                Create CMMS Ticket
              </button>
            </div>
            <div className="w-1/3 flex flex-col">
              <h3 className="text-xs font-bold text-gray-400 mb-2 uppercase">Causal Drivers (Δ SHAP)</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={MOCK_REPORT.causal_drivers} layout="vertical" margin={{ top: 0, right: 0, left: 10, bottom: 0 }}>
                  <XAxis type="number" hide />
                  <YAxis dataKey="feature" type="category" width={80} fontSize={9} stroke="#9ca3af" />
                  <Tooltip cursor={{ fill: '#374151' }} contentStyle={{ backgroundColor: '#1f2937', border: 'none', fontSize: 10 }} />
                  <Bar dataKey="delta_shap" barSize={12}>
                    {MOCK_REPORT.causal_drivers.map((entry, index) => (
                      <Cell key={index} fill={entry.delta_shap > 0 ? '#ef4444' : '#3b82f6'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Panel 5: RAG Chat */}
      <div className="h-48 border border-gray-800 rounded-lg bg-gray-900/50 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
          {chatHistory.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-lg p-3 text-sm ${msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-800 border border-gray-700'}`}>
                {msg.text}
              </div>
            </div>
          ))}
        </div>
        <form onSubmit={handleChat} className="border-t border-gray-800 p-3 flex gap-2">
          <input
            type="text"
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
            placeholder="Ask the SolarMind agent..."
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-4 outline-none focus:border-blue-500 transition-colors"
          />
          <button type="submit" className="bg-gray-100 text-gray-900 p-2 px-4 rounded font-bold hover:bg-white flex gap-2 items-center transition-colors">
            <Send size={16} /> Send
          </button>
        </form>
      </div>
    </div>
  );
}
