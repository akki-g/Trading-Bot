import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  ComposedChart
} from 'recharts';

const DataChart = ({ data, pair, interval }) => {
  if (!data || data.length === 0) {
    return <div className="loading">No chart data available</div>;
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    if (interval === '1m') {
      return date.toLocaleTimeString();
    } else {
      return date.toLocaleDateString();
    }
  };

  const formatTooltipDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const chartData = data.map(item => ({
    ...item,
    formattedTime: formatDate(item.time)
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="custom-tooltip" style={{
          backgroundColor: 'white',
          padding: '10px',
          border: '1px solid #ccc',
          borderRadius: '4px',
          boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
        }}>
          <p><strong>{formatTooltipDate(data.time)}</strong></p>
          <p style={{ color: '#8884d8' }}>Open: {data.open?.toFixed(5)}</p>
          <p style={{ color: '#82ca9d' }}>High: {data.high?.toFixed(5)}</p>
          <p style={{ color: '#ffc658' }}>Low: {data.low?.toFixed(5)}</p>
          <p style={{ color: '#ff7300' }}>Close: {data.close?.toFixed(5)}</p>
          {data.volume > 0 && <p style={{ color: '#8dd1e1' }}>Volume: {data.volume}</p>}
        </div>
      );
    }
    return null;
  };

  return (
    <div>
      <h3>Price Chart - {pair} ({interval})</h3>
      
      {/* OHLC Line Chart */}
      <div style={{ marginBottom: '30px' }}>
        <h4>OHLC Prices</h4>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="formattedTime" 
              tick={{ fontSize: 12 }}
              interval={Math.max(Math.floor(chartData.length / 10), 0)}
            />
            <YAxis 
              domain={['dataMin - 0.001', 'dataMax + 0.001']}
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => value.toFixed(5)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="open" 
              stroke="#8884d8" 
              strokeWidth={1}
              dot={false}
              name="Open"
            />
            <Line 
              type="monotone" 
              dataKey="high" 
              stroke="#82ca9d" 
              strokeWidth={1}
              dot={false}
              name="High"
            />
            <Line 
              type="monotone" 
              dataKey="low" 
              stroke="#ffc658" 
              strokeWidth={1}
              dot={false}
              name="Low"
            />
            <Line 
              type="monotone" 
              dataKey="close" 
              stroke="#ff7300" 
              strokeWidth={2}
              dot={false}
              name="Close"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Close Price with Volume */}
      <div style={{ marginBottom: '30px' }}>
        <h4>Close Price with Volume</h4>
        <ResponsiveContainer width="100%" height={400}>
          <ComposedChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="formattedTime" 
              tick={{ fontSize: 12 }}
              interval={Math.max(Math.floor(chartData.length / 10), 0)}
            />
            <YAxis 
              yAxisId="price"
              orientation="left"
              domain={['dataMin - 0.001', 'dataMax + 0.001']}
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => value.toFixed(5)}
            />
            <YAxis 
              yAxisId="volume"
              orientation="right"
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar 
              yAxisId="volume"
              dataKey="volume" 
              fill="#8dd1e1" 
              opacity={0.3}
              name="Volume"
            />
            <Line 
              yAxisId="price"
              type="monotone" 
              dataKey="close" 
              stroke="#ff7300" 
              strokeWidth={2}
              dot={false}
              name="Close Price"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Simple Candlestick representation */}
      <div>
        <h4>Price Range (High-Low)</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="formattedTime" 
              tick={{ fontSize: 12 }}
              interval={Math.max(Math.floor(chartData.length / 20), 0)}
            />
            <YAxis 
              domain={['dataMin - 0.001', 'dataMax + 0.001']}
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => value.toFixed(5)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar 
              dataKey="high" 
              fill="#82ca9d" 
              name="High"
            />
            <Bar 
              dataKey="low" 
              fill="#ffc658" 
              name="Low"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default DataChart;