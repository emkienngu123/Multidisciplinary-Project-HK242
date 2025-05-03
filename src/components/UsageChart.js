import React from 'react';
import { Bar } from 'react-chartjs-2';
import Card from './Card';

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const UsageChart = () => {

  const data = {
    labels: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    datasets: [
      {
        label: 'Số lần sử dụng',
        data: [12, 19, 3, 5, 2, 3, 7], 
        backgroundColor: '#D9D9D9',
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#D9D9D9',
        },
      },
      title: {
        display: true,
        text: 'Usage Statistics',
        color: '#D9D9D9',
      },
      tooltip: {
        bodyColor: '#D9D9D9',
        titleColor: '#D9D9D9',
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#D9D9D9',
        },
      },
      y: {
        ticks: {
          color: '#D9D9D9',
        },
      },
    },
  };

  return (
    <Card>
      <Bar data={data} options={options} />
    </Card>
  );
};

export default UsageChart;
