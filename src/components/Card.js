import React from 'react';

const Card = ({ children, className = '' }) => {
  return (
    <div className={`bg-[#484848] rounded-[30px] p-5 shadow-md m-5 ${className}`}>
      {children}
    </div>
  );
};

export default Card;
