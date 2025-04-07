import React, {useState} from "react";
import SettingsPopup from "./components/SettingsPopup";
import TopBar from "./components/Topbar";

const App = () => {
    const [showSettings, setShowSettings] = useState(false);

    const handleSettingsClick = () => {
        window.electronAPI.showSettings();
        setShowSettings(true);
    };

    return (
        <div className="min-h-screen flex flex-col bg-gray-950">
            <TopBar onSettingsClick={handleSettingsClick}/>

            <main className="flex-1 p-6">
                <div className="max-w-4xl mx-auto space-y-6">
                    <div
                        className="p-6 rounded-lg bg-gray-900 border border-purple-500/40 shadow-lg shadow-purple-500/10">
                        <h2 className="text-xl font-semibold text-purple-300 mb-3">Welcome to Your Application</h2>
                        <p className="text-gray-400">
                            This is an Electron app built with React and shadcn UI. Use the settings icon in the
                            topbar to configure your application preferences.
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div
                            className="p-5 rounded-lg bg-gray-900/80 border border-purple-700/30 hover:border-purple-500/40 transition-all">
                            <h3 className="text-lg font-medium text-purple-200 mb-2">Feature One</h3>
                            <p className="text-sm text-gray-400">Description of your first main feature goes
                                here.</p>
                        </div>

                        <div
                            className="p-5 rounded-lg bg-gray-900/80 border border-purple-700/30 hover:border-purple-500/40 transition-all">
                            <h3 className="text-lg font-medium text-purple-200 mb-2">Feature Two</h3>
                            <p className="text-sm text-gray-400">Description of your second main feature goes
                                here.</p>
                        </div>
                    </div>
                </div>
            </main>
            {/* Settings dialog */}
            <SettingsPopup open={showSettings} onOpenChange={setShowSettings}/>
        </div>
    );
};

export default App;