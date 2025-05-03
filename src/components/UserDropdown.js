import React, { useState, useRef, useEffect } from 'react';
import { FaUser } from 'react-icons/fa';
import { useNavigate } from 'react-router-dom';

export default function UserDropdown() {
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef(null);
  const navigate = useNavigate();

  const toggleOpen = () => setOpen(prev => !prev);
  const handleLogout = () => {
    localStorage.removeItem('jwtToken');
    navigate('/login');
  };

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative inline-block" ref={dropdownRef}>
      <button
        onClick={toggleOpen}
        className="bg-gray-800 p-4 rounded-full hover:bg-gray-700 cursor-pointer"
      >
        <FaUser className="text-white text-lg" />
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-40 bg-gray-800 rounded-md shadow-lg z-20">
          <button
            onClick={handleLogout}
            className="bg-gray-700 w-full text-left text-white px-4 py-2 hover:bg-gray-600 rounded-md"
          >
            Logout
          </button>
        </div>
      )}
    </div>
  );
}
