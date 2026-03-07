import React, { useEffect, useState } from 'react';
import { getTimeline, type TimelineEvent } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { AlertTriangle, Clock } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { formatDistanceToNow, parseISO } from 'date-fns';

export function FailureTimeline() {
    const [events, setEvents] = useState<TimelineEvent[]>([]);

    useEffect(() => {
        const fetchTimeline = async () => {
            const data = await getTimeline();
            setEvents(data);
        };
        fetchTimeline();
        const interval = setInterval(fetchTimeline, 5000);
        return () => clearInterval(interval);
    }, []);

    if (events.length === 0) {
        return (
            <Card className="col-span-1 border-white/10 bg-black/40 backdrop-blur-md">
                <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-medium flex items-center gap-2">
                        <Clock className="h-5 w-5 text-emerald-400" />
                        Predictive Timeline
                    </CardTitle>
                    <CardDescription>Estimated Time-to-Failure (TTF) forecasts</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex h-[250px] items-center justify-center text-sm text-gray-400">
                        No impending failures detected.
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="col-span-1 border-white/10 bg-black/40 backdrop-blur-md">
            <CardHeader className="pb-2">
                <CardTitle className="text-lg font-medium flex items-center gap-2">
                    <Clock className="h-5 w-5 text-blue-400" />
                    Predictive Timeline
                </CardTitle>
                <CardDescription>Estimated Time-to-Failure (TTF) forecasts</CardDescription>
            </CardHeader>
            <CardContent>
                <ScrollArea className="h-[250px] pr-4">
                    <div className="space-y-4 relative before:absolute before:inset-0 before:ml-5 before:-translate-x-px md:before:mx-auto md:before:translate-x-0 before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-slate-300 before:to-transparent">
                        {events.map((event, idx) => {
                            const timeStr = formatDistanceToNow(parseISO(event.predicted_failure_time), { addSuffix: true });
                            return (
                                <div key={idx} className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
                                    {/* Icon */}
                                    <div className="flex items-center justify-center w-10 h-10 rounded-full border border-white/10 bg-slate-900 group-[.is-active]:bg-red-500/20 text-slate-500 group-[.is-active]:text-red-400 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2">
                                        <AlertTriangle className="h-4 w-4" />
                                    </div>
                                    {/* Card */}
                                    <div className="w-[calc(100%-4rem)] md:w-[calc(50%-2.5rem)] p-4 rounded-lg border border-white/10 bg-slate-800/50 shadow">
                                        <div className="flex items-center justify-between mb-1">
                                            <div className="font-semibold text-slate-100">{event.inverter_id}</div>
                                            <time className="font-mono text-xs font-medium text-amber-500">{timeStr}</time>
                                        </div>
                                        <div className="text-xs text-slate-400 flex flex-col gap-1 mt-2">
                                            <span><strong>Type:</strong> <Badge variant="outline" className="text-xs">{event.failure_type.replace('_', ' ')}</Badge></span>
                                            <span><strong>Risk Score:</strong> {(event.risk_score * 100).toFixed(1)}%</span>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}
