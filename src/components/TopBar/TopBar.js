import { FaBars} from "react-icons/fa";
import NotificationDropdown from "../NotificationDropdown";
import MicCapture from "../MicCapture";
import UserDropdown from "../UserDropdown";
const TopBar = () => {
  return (
    <div className="bg-black flex justify-between items-center p-4">
      {/* Search Bar */}
      <div className="flex items-center bg-gray-800 text-white px-4 py-2 rounded-full w-1/3">
        <FaBars className="mr-2" />
        <input
          type="text"
          placeholder="Type something"
          className="bg-transparent focus:outline-none flex-grow"
        />
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width={"20px"}
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={2}
          stroke="currentColor"
          className="w-5 h-5 text-gray-400"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M21 21l-4.35-4.35m0 0a7.5 7.5 0 1 0-10.6 0 7.5 7.5 0 0 0 10.6 0z"
          />
        </svg>
      </div>
      {/* Icons */}
      <div className="flex gap-6">
        <MicCapture />
        <NotificationDropdown />
        <UserDropdown />
      </div>
    </div>
  );
};

export default TopBar;
