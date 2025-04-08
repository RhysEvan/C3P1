import React, {useEffect, useState} from "react";
import {Button} from "@/components/ui/button";
import {Dialog, DialogClose, DialogContent, DialogHeader, DialogTitle} from "@/components/ui/dialog";

const SettingsPopup = ({open, onOpenChange}) => {
    const [settings, setSettings] = useState({
        option1: "",
        option2: [],
        option3: "option1",
        option4: "value1",
    });
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        if (open) {
            readSettings();
        }
    }, [open]);

    const readSettings = async () => {
        try {
            setLoading(true);
            const loadedSettings = await window.electronAPI.getSettings();
            setSettings({
                option1: loadedSettings.option1 || "",
                option2: loadedSettings.option2 || [],
                option3: loadedSettings.option3 || "option1",
                option4: loadedSettings.option4 || "value1",
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

    const handleMultiSelectChange = (field, value) => {
        setSettings(prev => ({
            ...prev,
            [field]: prev[field].includes(value)
                ? prev[field].filter(item => item !== value)
                : [...prev[field], value]
        }));
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent
                className="bg-gray-800 text-gray-100 shadow-lg rounded-lg w-[20vw] min-w-[300px] max-w-[400px] border border-gray-300">
                <DialogHeader>
                    <DialogTitle className="text-2xl font-semibold text-purple-500">Settings</DialogTitle>
                </DialogHeader>
                {loading ? (
                    <div className="py-4">Loading settings...</div>
                ) : (
                    <div className="py-4 space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-50">Option 1</label>
                            <input
                                type="text"
                                value={settings.option1}
                                onChange={(e) => handleChange("option1", e.target.value)}
                                placeholder="Enter value"
                                className="mt-1 block w-full rounded-md border border-gray-300 bg-gray-800 p-2 text-gray-50"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-50">Option 2</label>
                            <div className="flex flex-wrap mt-1">
                                {["option1", "option2", "option3"].map(option => (
                                    <label key={option} className="flex items-center mr-4">
                                        <input
                                            type="checkbox"
                                            checked={settings.option2.includes(option)}
                                            onChange={() => handleMultiSelectChange("option2", option)}
                                            className="mr-2 accent-purple-600"
                                        />
                                        <span className="text-sm text-gray-50">{option}</span>
                                    </label>
                                ))}
                            </div>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-50">Option 3</label>
                            <div className="mt-1">
                                {["option1", "option2", "option3"].map(option => (
                                    <label key={option} className="flex items-center mr-4">
                                        <input
                                            type="radio"
                                            name="option3"
                                            value={option}
                                            checked={settings.option3 === option}
                                            onChange={(e) => handleChange("option3", e.target.value)}
                                            className="mr-2 accent-purple-600"
                                        />
                                        <span className="text-sm text-gray-50">{option}</span>
                                    </label>
                                ))}
                            </div>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-50">Option 4</label>
                            <select
                                value={settings.option4}
                                onChange={(e) => handleChange("option4", e.target.value)}
                                className="mt-1 block w-full rounded-md border border-gray-300 bg-gray-800 p-2 text-gray-50"
                            >
                                <option value="value1">Value 1</option>
                                <option value="value2">Value 2</option>
                                <option value="value3">Value 3</option>
                            </select>
                        </div>
                    </div>
                )}
                <div className="mt-4 flex justify-end space-x-2">
                    <Button
                        onClick={handleSaveSettings}
                        className="bg-background text-purple-500 border border-purple-700 hover:bg-purple-100"
                    >
                        Save
                    </Button>
                    <DialogClose asChild>
                        <Button
                            className="bg-background text-purple-500 border border-purple-300 hover:bg-purple-100"
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