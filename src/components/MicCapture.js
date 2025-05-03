import React, { useState, useEffect } from "react";
import { FaMicrophone, FaStop } from "react-icons/fa";
import MicRecorder from "mic-recorder-to-mp3";
import { voiceCommand, setLightIntensity, setFanSpeed, setAutoAdjust, getLightIntensity, getFanSpeed } from "../api";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const recorder = new MicRecorder({ bitRate: 128 });

export default function MicCapture() {
  const [isRecording, setIsRecording] = useState(false);
  const [fan, setFan] = useState(0);
  const [light, setLight] = useState(0);

  // load initial values once
  useEffect(() => {
    (async () => {
      try {
        
        const [initialFan, initialLight] = await Promise.all([
          getFanSpeed(),
          getLightIntensity()
        ]);
        setFan(initialFan);
        setLight(initialLight);
      } catch (err) {
        const token = localStorage.getItem('jwtToken');
        console.error('Failed to load initial device state', err);
      }
    })();
  }, []);

  const handleCommand = async (command) => {
    try {
      let msg;
      const cmd = command.toLowerCase();
      if (cmd.includes("increase light")) {
        const newVal = Math.min(light + 10, 100);
        await setLightIntensity(newVal);
        setLight(newVal);
        msg = "Light increased successfully";
      } else if (cmd.includes("decrease light")) {
        const newVal = Math.max(light - 10, 0);
        await setLightIntensity(newVal);
        setLight(newVal);
        msg = "Light decreased successfully";
      } else if (cmd.includes("increase fan")) {
        const newVal = Math.min(fan + 10, 100);
        await setFanSpeed(newVal);
        setFan(newVal);
        msg = "Fan speed increased successfully";
      } else if (cmd.includes("decrease fan")) {
        const newVal = Math.max(fan - 10, 0);
        await setFanSpeed(newVal);
        setFan(newVal);
        msg = "Fan speed decreased successfully";
      } else if (cmd.includes("auto on")) {
        await setAutoAdjust(true);
        msg = "Auto adjust enabled";
      } else if (cmd.includes("auto off")) {
        await setAutoAdjust(false);
        msg = "Auto adjust disabled";
      } else {
        msg = `Unknown command: ${command}`;
      }
      toast.success(msg);
    } catch (err) {
      console.error(err);
      toast.error(`Failed: ${err.message}`);
    }
  };

  const startRecording = () => {
    recorder
      .start()
      .then(() => setIsRecording(true))
      .catch(e => toast.error("Cannot start recording"));
  };

  const stopRecording = () => {
    recorder
      .stop()
      .getMp3()
      .then(async ([buffer, blob]) => {
        setIsRecording(false);
        const mp3File = new File(buffer, "voice.mp3", {
          type: blob.type,
          lastModified: Date.now()
        });
        try {
          const command = await voiceCommand(mp3File);
          await handleCommand(command);
        } catch (err) {
          console.error("Voice command error:", err);
          toast.error(`Error: ${err.message}`);
        }
      })
      .catch(e => toast.error("Cannot stop recording"));
  };
  return (
    <div className="flex flex-col items-center">
      <button
        onClick={isRecording ? stopRecording : startRecording}
        className={`p-4 rounded-full focus:outline-none cursor-pointer \
          ${isRecording ? "bg-red-600 hover:bg-red-500" : "bg-gray-800 hover:bg-gray-700"} \
          transition-colors duration-200 text-white`}
      >
        {isRecording ? <FaStop size={24} /> : <FaMicrophone size={24} />}
      </button>
      {/* Toast container for notifications */}
      <ToastContainer position="top-right" autoClose={5000} hideProgressBar={false} newestOnTop={false} closeOnClick rtl={false} pauseOnFocusLoss draggable pauseOnHover />
    </div>
  );
}
