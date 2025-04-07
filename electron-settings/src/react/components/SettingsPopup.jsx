import React, {useEffect, useState} from "react";
import {Button} from "@/components/ui/button";
import {Dialog, DialogClose, DialogContent, DialogHeader, DialogTitle,} from "@/components/ui/dialog";

const SettingsPopup = ({open, onOpenChange}) => {
    const [settings, setSettings] = useState({
        option1: "",
        option2: false,
    });
    const [loading, setLoading] = useState(true);

    // Load settings when dialog opens
    useEffect(() => {
        if (open) {
            loadSettings();
        }
    }, [open]);

    const loadSettings = async () => {
        try {
            setLoading(true);
            const loadedSettings = await window.electronAPI.getSettings();
            setSettings({
                option1: loadedSettings.option1 || "",
                option2: loadedSettings.option2 || false,
            });
        } catch (error) {
            console.error("Failed to load settings:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleSaveSettings = async () => {
        try {
            await window.electronAPI.saveSettings(settings);
            onOpenChange(false);
        } catch (error) {
            console.error("Failed to save settings:", error);
        }
    };

    const handleChange = (field, value) => {
        setSettings(prev => ({
            ...prev,
            [field]: value
        }));
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent
                className="bg-gray-200 text-gray-900 shadow-lg rounded-lg w-[20vw] min-w-[200px] max-w-[400px] border border-gray-300">
                <DialogHeader>
                    <DialogTitle className="text-2xl font-semibold text-gray-800">Settings</DialogTitle>
                </DialogHeader>
                {loading ? (
                    <div className="py-4">Loading settings...</div>
                ) : (
                    <div className="py-4 space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700">
                                Option 1
                            </label>
                            <input
                                type="text"
                                value={settings.option1}
                                onChange={(e) => handleChange("option1", e.target.value)}
                                placeholder="Enter value"
                                className="mt-1 block w-full rounded-md border border-gray-300 bg-white p-2 text-gray-900"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700">
                                Option 2
                            </label>
                            <div className="flex items-center mt-1">
                                <input
                                    type="checkbox"
                                    checked={settings.option2}
                                    onChange={(e) => handleChange("option2", e.target.checked)}
                                    className="mr-2 accent-purple-600"
                                />
                                <span className="text-sm text-gray-800">Enable Option</span>
                            </div>
                        </div>
                    </div>
                )}
                <div className="mt-4 flex justify-end space-x-2">
                    <Button
                        onClick={handleSaveSettings}
                        className="bg-white text-white border border-purple-700 hover:bg-purple-100"
                    >
                        Save
                    </Button>
                    <DialogClose asChild>
                        <Button
                            variant="outline"
                            className="bg-white text-gray-800 border-purple-300 hover:bg-purple-100"
                        >
                            Cancel
                        </Button>
                    </DialogClose>
                </div>
            </DialogContent>
        </Dialog>
    );
};

export default SettingsPopup;