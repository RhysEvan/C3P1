import React, {useState} from "react";
import {Settings} from "lucide-react";
import {Button} from "@/components/ui/button";
import SettingsPopup from "./components/SettingsPopup";

const App = () => {
    const [showSettings, setShowSettings] = useState(false);

    const handleSettingsClick = () => {
        window.electronAPI.showSettings();
        setShowSettings(true);
    };

    return (
        <div id="app" className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-950 to-gray-900 text-gray-100">
            <div className="flex justify-between items-center p-4 border-b border-purple-700/30 bg-gray-950/70">
                <h1 className="text-2xl font-bold text-purple-300">Electron Shadcn Settings</h1>
                <Button
                    variant="outline"
                    size="icon"
                    onClick={handleSettingsClick}
                    aria-label="Settings"
                    className="!p-3 px-4 border-purple-700 border-2 bg-purple-900/20 hover:bg-purple-900/30 hover:shadow-md hover:shadow-purple-500/30 transition-all duration-300"
                >
                    <Settings className="h-5 w-5 text-purple-400 hover:text-purple-300"/>
                </Button>

            </div>

            <div className="p-6">
                <div className="max-w-2xl mx-auto p-4 rounded-lg bg-gray-800/50 border border-purple-500/20">
                    <p>Welcome to the Electron app with shadcn UI. Click the settings icon to open the popup.</p>
                </div>
            </div>

            {/* Render the settings dialog */}
            <SettingsPopup open={showSettings} onOpenChange={setShowSettings}/>
        </div>
    );
};

export default App;