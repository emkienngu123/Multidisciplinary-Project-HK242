// src/App.js
import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import "./App.css";

import Sidebar from "./components/SideBar/SideBar";
import Topbar  from "./components/TopBar/TopBar";
import LoginPage from "./components/LoginPage";
import ProtectedRoute from "./components/ProtectedRoute";

import Home     from "./pages/Home";
import Security from "./pages/Security";
import CCTV     from "./pages/CCTV";
import Settings from "./pages/Settings";
import Help     from "./pages/Help";

function AppContent() {
  const location = useLocation();
  const isLoginPage = location.pathname === "/login";

  return (
    <div className="app-container">
      {/* Chỉ show Sidebar nếu không phải login */}
      {!isLoginPage && <Sidebar />}

      <div className="main-content">
        {/* Chỉ show Topbar nếu không phải login */}
        {!isLoginPage && <Topbar />}

        <div className="page-content">
          <Routes>
            {/* Public */}
            <Route path="/login" element={<LoginPage />} />

            {/* Protected routes */}
            <Route 
              path="/" 
              element={
                <ProtectedRoute>
                  <Home />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/security" 
              element={
                <ProtectedRoute>
                  <Security />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/cctv" 
              element={
                <ProtectedRoute>
                  <CCTV />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/settings" 
              element={
                <ProtectedRoute>
                  <Settings />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/help" 
              element={
                <ProtectedRoute>
                  <Help />
                </ProtectedRoute>
              } 
            />
          </Routes>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  );
}
