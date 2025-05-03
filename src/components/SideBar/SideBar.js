import React from "react";
import { Link } from "react-router-dom";
import { FaHome, FaShieldAlt, FaVideo, FaCog, FaQuestionCircle } from "react-icons/fa";

function Sidebar() {
  return (
    <div className="w-[250px] bg-black text-white flex flex-col py-5">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold">AIoT Home</h2>
      </div>
      <ul className="list-none p-0 m-0">
        <li className="my-3.5">
          <Link
            to="/"
            className="flex items-center text-white no-underline px-5 py-2 rounded-lg transition-colors duration-300 hover:bg-gray-800"
          >
            <FaHome className="mr-2.5 text-2xl" />
            <span className="text-xl">Home</span>
          </Link>
        </li>
        <li className="my-3.5">
          <Link
            to="/security"
            className="flex items-center text-white no-underline px-5 py-2 rounded-lg transition-colors duration-300 hover:bg-gray-800"
          >
            <FaShieldAlt className="mr-2.5 text-2xl" />
            <span className="text-xl">Security</span>
          </Link>
        </li>
        <li className="my-3.5">
          <Link
            to="/cctv"
            className="flex items-center text-white no-underline px-5 py-2 rounded-lg transition-colors duration-300 hover:bg-gray-800"
          >
            <FaVideo className="mr-2.5 text-2xl" />
            <span className="text-xl">CCTV</span>
          </Link>
        </li>
        <li className="my-3.5">
          <Link
            to="/settings"
            className="flex items-center text-white no-underline px-5 py-2 rounded-lg transition-colors duration-300 hover:bg-gray-800"
          >
            <FaCog className="mr-2.5 text-2xl" />
            <span className="text-xl">Settings</span>
          </Link>
        </li>
        <li className="my-3.5">
          <Link
            to="/help"
            className="flex items-center text-white no-underline px-5 py-2 rounded-lg transition-colors duration-300 hover:bg-gray-800"
          >
            <FaQuestionCircle className="mr-2.5 text-2xl" />
            <span className="text-xl">Help</span>
          </Link>
        </li>
      </ul>
    </div>
  );
}

export default Sidebar;
