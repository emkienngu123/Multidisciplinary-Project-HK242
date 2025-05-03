// src/components/ProtectedRoute.jsx
import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';

export default function ProtectedRoute({ children }) {

  const isAuth = Boolean(localStorage.getItem('jwtToken'));  
  const location = useLocation();

  if (!isAuth) {

    return <Navigate to="/login" replace state={{ from: location }} />;
  }
  return children;
}
