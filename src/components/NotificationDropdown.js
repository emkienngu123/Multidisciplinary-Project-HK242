import React, { useState } from "react";
import { FaBell, FaLightbulb, FaUser } from "react-icons/fa"; // ví dụ dùng react-icons

function NotificationDropdown() {

  const [open, setOpen] = useState(false);

  const handleToggle = () => {
    setOpen(!open);
  };

  return (
    <div className="relative inline-block">
      {/* Nút Thông báo */}
      <button
        onClick={handleToggle}
        className="p-4 rounded-full bg-gray-800 text-white text-large hover:bg-gray-700 cursor-pointer focus:outline-none"
      >
        <FaBell size={20} />
      </button>

      {open && (
        <div
          className="
            absolute 
            right-0 
            mt-2 
            w-72 
            bg-gray-700 
            text-white 
            rounded-md 
            shadow-lg 
            z-10
          "
        >
          {/* Header */}
          <div className="px-4 py-3 border-b border-gray-600">
            <h3 className="text-lg font-semibold">Notifications</h3>
          </div>

          {/* Nội dung thông báo */}
          <div className="p-4 space-y-4">
            {/* Thông báo 1 */}
            <div className="flex items-start">
              <div className="p-2 bg-gray-600 rounded-full mr-3">
                <FaLightbulb />
              </div>
              <div className="text-sm leading-tight">
                <p className="font-medium">Low light condition. The lights are turned on!</p>
                <span className="text-xs text-gray-300">5:45</span>
              </div>
            </div>

            {/* Thông báo 2 */}
            <div className="flex items-start">
              <div className="p-2 bg-gray-600 rounded-full mr-3">
                <FaUser />
              </div>
              <div className="text-sm leading-tight">
                <p className="font-medium">Someone has just logged in!</p>
                <span className="text-xs text-gray-300">4:20</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default NotificationDropdown;
