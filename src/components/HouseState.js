// src/components/HouseState/HouseState.jsx
import React, { useState, useEffect } from 'react';

import { getTemperature, getHumidity, getMovement, getLight, getFanSpeed, setFanSpeed, getLightIntensity, setLightIntensity} from '../api';



export default function HouseState() {
  const [sensors, setSensors] = useState({
    temperature: 'Loading...',
    brightness:  'Loading...',
    airQuality:  'Loading...',
    movement:    'Loading...',
  });

  useEffect(() => {
    const token = localStorage.getItem('jwtToken');
    if (!token) {
      window.location.href = '/login';
      return;
    }
    let intervalId;
    // fetch latest sensor data
    const fetchData = async () => {
      try {
        const [tempRes, humidityRes, lightRes, movementRes] = await Promise.all([
          getTemperature(1),
          getHumidity(1),
          getLight(1),
          getMovement(1)
        ]);
        setSensors({
          temperature: `${tempRes.value[0]} °C`,  
          brightness:  `${lightRes.value[0]} %`,  
          airQuality:  `${humidityRes.value[0]} %`, 
          movement:    movementRes.value[0] === 1 ? 'Detected' : 'Not Detected'
        });
      } catch (err) {
        console.error('Error fetching sensor data:', err);
      }
    };

    // initial load
    fetchData();
    // poll every 5 seconds
    intervalId = setInterval(fetchData, 5000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="bg-[#484848] rounded-lg shadow p-4 mb-5 w-1/2">
      <div className="flex items-center space-x-2 mb-4">
        <div className="bg-[#727272] text-[#D9D9D9] rounded-full w-12 h-12 flex items-center justify-center">
          <i className="fa-solid fa-signal inline-block text-2xl"></i>
        </div>
        <h2 className="text-xl font-bold text-white">House state</h2>
      </div>
      <div className="grid grid-cols-2 gap-4 text-[#D9D9D9]">
        {/* Nhiệt độ */}
        <div className="flex flex-col items-center">
          <div className="bg-[#0087FC] rounded-full w-10 h-10 flex items-center justify-center mb-1">
            <i className="fa-solid fa-temperature-three-quarters text-white"></i>
          </div>
          <p>{sensors.temperature}</p>
        </div>
        {/* Độ sáng */}
        <div className="flex flex-col items-center">
          <div className="bg-[#FAFF97] rounded-full w-10 h-10 flex items-center justify-center mb-1">
            <i className="fa-solid fa-lightbulb text-black"></i>
          </div>
          <p>{sensors.brightness}</p>
        </div>
        {/* Không khí */}
        <div className="flex flex-col items-center">
          <div className="bg-[#00A71F] rounded-full w-10 h-10 flex items-center justify-center mb-1">
            <i className="fa-solid fa-droplet text-white"></i>
          </div>
          <p>{sensors.airQuality}</p>
        </div>
        {/* Chuyển động */}
        <div className="flex flex-col items-center">
          <div className="bg-[#8BCBDB] rounded-full w-10 h-10 flex items-center justify-center mb-1">
            <i className="fa-solid fa-person-walking text-[#FFFB00]"></i>
          </div>
          <p>{sensors.movement}</p>
        </div>
      </div>
    </div>
  );
}


function ManualAdjustment() {
  const [fan, setFan] = useState(0);
  const [light, setLight] = useState(0);

  useEffect(() => {
    const token = localStorage.getItem('jwtToken');
    if (!token) {

      window.location.href = '/login';
      return;
    }

    const fetchInitial = async () => {
      const [f, l] = await Promise.all([getFanSpeed(), getLightIntensity()]);
      setFan(f);
      setLight(l);
    };

    fetchInitial();

    // Polling: fetch lại mỗi 2s
    const intervalId = setInterval(async () => {
      const [f, l] = await Promise.all([getFanSpeed(), getLightIntensity()]);
      setFan(f);
      setLight(l);
    }, 2000);

    return () => clearInterval(intervalId);
  }, []);

  const handleFanChange = async e => {
    const v = +e.target.value;
    setFan(v);
    await setFanSpeed(v);
  };

  const handleLightChange = async e => {
    const v = +e.target.value;
    setLight(v);
    await setLightIntensity(v);
  };


  const makeGradient = (value, color) =>
    `linear-gradient(to right, ${color} 0%, ${color} ${value}%, #2d2d2d ${value}%, #2d2d2d 100%)`;

  return (
    <div className="bg-[#484848] rounded-lg shadow p-4 w-1/2 space-y-6">
      {/* Fans */}
      <div>
        <h3 className="flex items-center space-x-2 text-white mb-2">
          <div className="bg-[#0087FC] rounded-full w-8 h-8 flex items-center justify-center">
            <i className="fa-solid fa-fan text-white" />
          </div>
          <span>Fans</span>
        </h3>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-[#D9D9D9]">Min (Off)</span>
          <input
            type="range"
            min="0"
            max="100"
            value={fan}
            onChange={handleFanChange}
            className="w-full h-2 rounded-lg appearance-none focus:outline-none"
            style={{
              background: makeGradient(fan, '#3b82f6')  
            }}
          />
          <span className="text-sm text-[#D9D9D9]">Max</span>
        </div>
      </div>

      {/* Lights */}
      <div>
        <h3 className="flex items-center space-x-2 text-white mb-2">
          <div className="bg-[#FAFF97] rounded-full w-8 h-8 flex items-center justify-center">
            <i className="fa-solid fa-lightbulb text-black" />
          </div>
          <span>Lights</span>
        </h3>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-[#D9D9D9]">Min (Off)</span>
          <input
            type="range"
            min="0"
            max="100"
            value={light}
            onChange={handleLightChange}
            className="w-full h-2 rounded-lg appearance-none focus:outline-none"
            style={{
              background: makeGradient(light, '#fbbf24') 
            }}
          />
          <span className="text-sm text-[#D9D9D9]">Max</span>
        </div>
      </div>
    </div>
  );
}


export { HouseState, ManualAdjustment };
