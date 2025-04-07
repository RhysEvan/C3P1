import React from "react";
import { Settings, Bell, Moon, Search, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";

const TopBar = ({ onSettingsClick }) => {
  const buttonClasses = "border-purple-700 border-2 bg-purple-900/20 hover:bg-purple-900/30 hover:shadow-md hover:shadow-purple-500/30 transition-all duration-300";

  return (
    <div className="flex justify-between items-center p-4 border-b border-purple-700/30 bg-gray-950/70">
      <div className="flex items-center">
        <h1 className="text-2xl font-bold text-purple-300">Electron Shadcn Settings</h1>
      </div>

      <div className="flex items-center space-x-3">
        {/* Search - hidden on mobile */}
        <div className="relative mr-1 hidden md:block">
          <Input
            className="w-[180px] lg:w-[240px] h-9 bg-gray-800/50 border-purple-700/40 focus:border-purple-500/70"
            placeholder="Search..."
          />
          <Search className="absolute right-2 top-2.5 h-4 w-4 text-purple-400" />
        </div>

        {/* Theme toggle */}
        <Button
          variant="outline"
          size="icon"
          aria-label="Toggle theme"
          className={`!p-3 ${buttonClasses}`}
        >
          <Moon className="h-5 w-5 text-purple-400 hover:text-purple-300" />
        </Button>

        {/* Notifications */}
        <Button
          variant="outline"
          size="icon"
          aria-label="Notifications"
          className={`!p-3 ${buttonClasses}`}
        >
          <Bell className="h-5 w-5 text-purple-400 hover:text-purple-300" />
        </Button>

        {/* Help */}
        <Button
          variant="outline"
          size="icon"
          aria-label="Help"
          className={`!p-3 ${buttonClasses}`}
        >
          <HelpCircle className="h-5 w-5 text-purple-400 hover:text-purple-300" />
        </Button>

        {/* Settings - with functionality */}
        <Button
          variant="outline"
          size="icon"
          onClick={onSettingsClick}
          aria-label="Settings"
          className={`!p-3 ${buttonClasses}`}
        >
          <Settings className="h-5 w-5 text-purple-400 hover:text-purple-300" />
        </Button>

        {/* User Profile */}
        <Button
          variant="outline"
          className={`ml-1 px-3 py-1 flex items-center space-x-2 ${buttonClasses}`}
        >
          <span className="hidden sm:inline text-sm text-purple-300">User</span>
        </Button>
      </div>
    </div>
  );
};

export default TopBar;