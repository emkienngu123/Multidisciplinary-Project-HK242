import React, { useState, useRef, useEffect } from "react";

function CameraFaceRecognition() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);

  // Bắt camera
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = mediaStream;
      setStream(mediaStream);
    } catch (err) {
      console.error("Không thể truy cập camera:", err);
    }
  };

  // Dừng camera
  const stopCamera = () => {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
  };


  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, []);


  return (
    <div className="min-h-screen flex flex-col items-center">
      {/* Video element */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="border border-gray-300 rounded-md"
        style={{ width: "640px", height: "480px", backgroundColor: "#000" }}
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <div className="mt-4 flex gap-2">
        <button
          onClick={startCamera}
          className="px-4 py-2 bg-gray-800 text-white rounded hover:bg-gray-700 transition-colors duration-300"
        >
          Start Camera
        </button>
        <button
          onClick={stopCamera}
          className="px-4 py-2 bg-gray-800 text-white rounded hover:bg-gray-700 transition-colors duration-300"
        >
          Stop Camera
        </button>
      </div>
    </div>
  );

}

export default CameraFaceRecognition;
