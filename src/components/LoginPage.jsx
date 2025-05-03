import React, { useState, useRef, useEffect } from 'react';
import { signIn, faceSignIn } from '../api';
import { useNavigate } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

export default function LoginPage() {
  const [mode, setMode] = useState('password');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (mode === 'face') {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          streamRef.current = stream;
          videoRef.current.srcObject = stream;
        })
        .catch(() => {
          toast.error('Bạn cần cho phép truy cập camera để đăng nhập bằng khuôn mặt.');
        });
    } else {
      streamRef.current?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    return () => streamRef.current?.getTracks().forEach(t => t.stop());
  }, [mode]);

  const handlePasswordLogin = async e => {
    e.preventDefault();
    try {
      await signIn(username, password);
      toast.success('Login successful!');
      navigate('/');
    } catch (err) {
      toast.error('Login failed. Please check your username and password.');
    }
  };

  const handleFaceLogin = async () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob(async blob => {
      if (!blob) return toast.error('Lấy hình ảnh thất bại, thử lại.');
      const form = new FormData();
      form.append('image', blob, 'face.jpg');
      try {
        const data = await faceSignIn(form);
        localStorage.setItem('jwtToken', data.accessToken);
        toast.success('Face login successful!');
        navigate('/');
      } catch (err) {
        toast.error('Face login failed. Please try again');
      }
    }, 'image/jpeg', 0.9);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-black">
      <div className="bg-gray-800 rounded-2xl shadow-xl w-full max-w-md p-8">
        <h1 className="text-2xl font-bold text-white text-center mb-6">Welcome to AIoT Home</h1>
        <div className="flex justify-center mb-6">
          <button
            className={`px-4 py-2 mx-2 rounded-full ${mode === 'password' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-600'}`}
            onClick={() => setMode('password')}
          >Username & Password</button>
          <button
            className={`px-4 py-2 mx-2 rounded-full ${mode === 'face' ? 'bg-green-600 text-white' : 'bg-gray-200 text-gray-600'}`}
            onClick={() => setMode('face')}
          >Face Recognition</button>
        </div>
        {mode === 'password' ? (
          <form onSubmit={handlePasswordLogin} className="space-y-4">
            <div>
              <label className="block text-white">Username</label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
                className="w-full mt-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-white text-white"
                required
              />
            </div>
            <div>
              <label className="block text-white">Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                className="w-full mt-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-white text-white"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition"
            >Log In</button>
          </form>
        ) : (
          <div className="flex flex-col items-center space-y-4">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full rounded-lg border border-gray-600"
            />
            <button
              onClick={handleFaceLogin}
              className="w-full py-2 bg-green-600 text-white rounded-lg hover:bg-green-500 transition"
            >Authenticate Face</button>
          </div>
        )}
      </div>
      <ToastContainer position="top-right" autoClose={5000} hideProgressBar={false} newestOnTop={false} closeOnClick pauseOnHover draggable />
    </div>
  );
}
