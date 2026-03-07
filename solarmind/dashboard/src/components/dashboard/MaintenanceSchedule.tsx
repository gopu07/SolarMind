import React, { useEffect, useState } from 'react';
import { getMaintenanceSchedule, type MaintenanceTask } from '@/services/api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Wrench, CalendarClock, AlertCircle } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { format, parseISO } from 'date-fns';

export function MaintenanceSchedule() {
    const [tasks, setTasks] = useState<MaintenanceTask[]>([]);

    useEffect(() => {
        const fetchTasks = async () => {
            const data = await getMaintenanceSchedule();
            setTasks(data);
        };
        fetchTasks();
        const interval = setInterval(fetchTasks, 5000);
        return () => clearInterval(interval);
    }, []);

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'CRITICAL': return 'bg-red-500/20 text-red-400 border-red-500/50';
            case 'HIGH': return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
            case 'MEDIUM': return 'bg-amber-500/20 text-amber-400 border-amber-500/50';
            default: return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
        }
    };

    if (tasks.length === 0) {
        return (
            <Card className="col-span-1 border-white/10 bg-black/40 backdrop-blur-md">
                <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-medium flex items-center gap-2">
                        <Wrench className="h-5 w-5 text-emerald-400" />
                        Maintenance Schedule
                    </CardTitle>
                    <CardDescription>Prioritized dispatch recommendations</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex h-[250px] items-center justify-center text-sm text-gray-400">
                        No pending maintenance tasks.
                    </div>
                </CardContent>
            </Card>
        );
    }

    return (
        <Card className="col-span-1 border-white/10 bg-black/40 backdrop-blur-md flex flex-col h-full">
            <CardHeader className="pb-2">
                <CardTitle className="text-lg font-medium flex items-center gap-2">
                    <Wrench className="h-5 w-5 text-indigo-400" />
                    Maintenance Schedule
                </CardTitle>
                <CardDescription>Prioritized dispatch recommendations</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden">
                <ScrollArea className="h-[250px] pr-4">
                    <div className="space-y-3">
                        {tasks.map((task) => (
                            <div
                                key={task.maintenance_id}
                                className={`p-3 rounded-lg border bg-black/40 ${getPriorityColor(task.priority)}`}
                            >
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2">
                                        <span className="font-bold">{task.inverter_id}</span>
                                        <Badge variant="outline" className={getPriorityColor(task.priority)}>{task.priority}</Badge>
                                    </div>
                                    <div className="flex items-center gap-1 text-xs opacity-80">
                                        <CalendarClock className="h-3 w-3" />
                                        {format(parseISO(task.recommended_time), 'MMM d, HH:mm')}
                                    </div>
                                </div>

                                <div className="flex items-start gap-2 text-sm mt-1">
                                    <AlertCircle className="h-4 w-4 mt-0.5 opacity-70 shrink-0" />
                                    <p className="opacity-90 leading-snug">{task.recommended_action}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}
