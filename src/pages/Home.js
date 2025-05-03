// src/pages/Home.js
import React from "react";
import CameraFeed from "../components/CameraFeed/CameraFeed";
import UsageChart from '../components/UsageChart';
import { HouseState, ManualAdjustment } from '../components/HouseState';
function Home() {
  return (
    <>
      <CameraFeed />     
      <UsageChart />
      <div className="flex flex-col items-center">
        <HouseState />
        <ManualAdjustment />
      </div>
    </>
  );
}

export default Home;
