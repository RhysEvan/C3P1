import React, {useState} from 'react';
import TopBar from './components/TopBar';
import SettingsPopup from './components/SettingsPopup';

const App = () => {
    const [showSettings, setShowSettings] = useState(false);
    const [pythonResult, setPythonResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleSettingsClick = () => {
        window.electronAPI.showSettings();
        setShowSettings(true);
    };

    const handleRunPython = async () => {
        try {
            setIsLoading(true);
            const result = await window.electronAPI.runPython({
                test: true,
                message: "Hello from React!"
            });
            console.log("Python script result:", result);
            setPythonResult(result);
        } catch (error) {
            console.error("Failed to run Python script:", error);
            setPythonResult({error: error.toString()});
        } finally {
            setIsLoading(false);
        }
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

                        <div className="mt-4">
                            <button
                                onClick={handleRunPython}
                                disabled={isLoading}
                                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-md transition-colors">
                                {isLoading ? 'Running Python...' : 'Run Python Test'}
                            </button>
                        </div>

                        {pythonResult && (
                            <div className="mt-4 p-4 bg-gray-800 rounded-md">
                                <h3 className="text-md font-medium text-purple-200 mb-2">Python Result:</h3>
                                <pre className="text-xs text-gray-300 overflow-auto max-h-60">
                                                {JSON.stringify(pythonResult, null, 2)}
                                            </pre>
                            </div>
                        )}
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