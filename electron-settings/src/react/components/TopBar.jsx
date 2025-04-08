import React from "react";
import {Bell, HelpCircle, Moon, Search, Settings} from "lucide-react";
import {Button} from "@/components/ui/button";
import {Avatar, AvatarFallback, AvatarImage} from "@/components/ui/avatar";
import {Input} from "@/components/ui/input";

const TopBar = ({onSettingsClick}) => {
    const buttonClasses = "border-purple-700 border-2 bg-purple-900/20 hover:bg-purple-900/50 hover:shadow-lg hover:shadow-purple-500/50 transition-all duration-300";

    return (
        <div className="flex justify-between items-center p-4 border-b border-purple-700/30 bg-gray-950/70">
            <div className="flex items-center">
                <h1 className="text-2xl font-bold text-purple-300">Electron Shadcn Settings</h1>
            </div>

            <div className="flex items-center space-x-3">
                <Button
                    variant="outline"
                    size="icon"
                    aria-label="Toggle theme"
                    className={`!p-3 ${buttonClasses}`}
                >
                    <Moon className="h-5 w-5 text-purple-400 hover:text-purple-950"/>
                </Button>

                <Button
                    variant="outline"
                    size="icon"
                    aria-label="Notifications"
                    className={`!p-3 ${buttonClasses}`}
                >
                    <Bell className="h-5 w-5 text-purple-400 hover:text-purple-950"/>
                </Button>

                <Button
                    variant="outline"
                    size="icon"
                    aria-label="Help"
                    className={`!p-3 ${buttonClasses}`}
                >
                    <HelpCircle className="h-5 w-5 text-purple-400 hover:text-purple-950"/>
                </Button>

                <Button
                    variant="outline"
                    size="icon"
                    onClick={onSettingsClick}
                    aria-label="Settings"
                    className={`!p-3 ${buttonClasses}`}
                >
                    <Settings className="h-5 w-5 text-purple-400 hover:text-white-200"/>
                </Button>

                <Button
                    variant="outline"
                    className={`ml-1 px-3 py-1 flex items-center space-x-2 ${buttonClasses}`}
                >
                    <Avatar className="h-6 w-6">
                        <AvatarImage src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExbWRnZHNpbXFmem1wZzdqNmp1Mzc1emxmOXZjcnU4cnd6Y3FqbHFrZyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/CAYVZA5NRb529kKQUc/giphy.gif" alt="Chad Avatar"/>
                        <AvatarFallback>U</AvatarFallback>
                    </Avatar>
                    <span className="hidden sm:inline text-sm text-purple-300">User</span>
                </Button>
            </div>
        </div>
    );
};

export default TopBar;