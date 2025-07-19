import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import DataFilters from './components/DataFilters';
import DataTable from './components/DataTable';
import DataChart from './components/DataChart';
import StatsSummary from './components/StatsSummary';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    pair: 'EURUSD',
    interval_type: 'daily',
    start_date: '',
    end_date: '',
    limit: 1000,
    page: 1
  });
  const [availablePairs, setAvailablePairs] = useState([]);
  const [availableIntervals, setAvailableIntervals] = useState([]);
  const [statistics, setStatistics] = useState(null);
  const [chartData, setChartData] = useState([]);
  const [activeView, setActiveView] = useState('table');

  useEffect(() => {
    fetchAvailablePairs();
    fetchAvailableIntervals();
  }, []);

  useEffect(() => {
    if (filters.pair && filters.interval_type) {
      fetchData();
      fetchStatistics();
      fetchChartData();
    }
  }, [filters]);

  const fetchAvailablePairs = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/pairs`);
      setAvailablePairs(response.data.pairs);
      if (response.data.pairs.length > 0 && !filters.pair) {
        setFilters(prev => ({ ...prev, pair: response.data.pairs[0] }));
      }
    } catch (err) {
      setError('Failed to fetch available pairs');
    }
  };

  const fetchAvailableIntervals = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/intervals`);
      setAvailableIntervals(response.data.intervals);
      // Set default interval if none selected and intervals are available
      if (response.data.intervals.length > 0 && !filters.interval_type) {
        const defaultInterval = response.data.intervals.find(i => 
          (typeof i === 'string' ? i : i.value) === 'daily'
        ) || response.data.intervals[0];
        const intervalValue = typeof defaultInterval === 'string' ? defaultInterval : defaultInterval.value;
        setFilters(prev => ({ ...prev, interval_type: intervalValue }));
      }
    } catch (err) {
      setError('Failed to fetch available intervals');
    }
  };

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value) params.append(key, value);
      });

      const response = await axios.get(`${API_BASE_URL}/data?${params}`);
      setData(response.data.data);
    } catch (err) {
      setError('Failed to fetch data: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchStatistics = async () => {
    try {
      const params = new URLSearchParams();
      if (filters.pair) params.append('pair', filters.pair);
      if (filters.interval_type) params.append('interval_type', filters.interval_type);
      if (filters.start_date) params.append('start_date', filters.start_date);
      if (filters.end_date) params.append('end_date', filters.end_date);

      const response = await axios.get(`${API_BASE_URL}/statistics?${params}`);
      setStatistics(response.data);
    } catch (err) {
      console.error('Failed to fetch statistics:', err);
    }
  };

  const fetchChartData = async () => {
    try {
      const params = new URLSearchParams();
      if (filters.pair) params.append('pair', filters.pair);
      if (filters.interval_type) params.append('interval_type', filters.interval_type);
      if (filters.start_date) params.append('start_date', filters.start_date);
      if (filters.end_date) params.append('end_date', filters.end_date);
      params.append('limit', '500');

      const response = await axios.get(`${API_BASE_URL}/chart-data?${params}`);
      setChartData(response.data.chart_data);
    } catch (err) {
      console.error('Failed to fetch chart data:', err);
    }
  };

  const handleFilterChange = (newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters, page: 1 }));
  };

  const handlePageChange = (newPage) => {
    setFilters(prev => ({ ...prev, page: newPage }));
  };

  return (
    <div className="App">
      <div className="container">
        <h1>Forex Data Visualization Dashboard</h1>
        
        <div className="card">
          <DataFilters
            filters={filters}
            availablePairs={availablePairs}
            availableIntervals={availableIntervals}
            onFilterChange={handleFilterChange}
          />
        </div>

        {statistics && (
          <div className="card">
            <StatsSummary statistics={statistics} />
          </div>
        )}

        <div className="card">
          <div className="view-selector">
            <button 
              className={`button ${activeView === 'table' ? 'active' : ''}`}
              onClick={() => setActiveView('table')}
            >
              Table View
            </button>
            <button 
              className={`button ${activeView === 'chart' ? 'active' : ''}`}
              onClick={() => setActiveView('chart')}
            >
              Chart View
            </button>
          </div>

          {error && <div className="error">{error}</div>}
          
          {loading && <div className="loading">Loading data...</div>}

          {!loading && activeView === 'table' && (
            <DataTable 
              data={data} 
              filters={filters}
              onPageChange={handlePageChange}
            />
          )}

          {!loading && activeView === 'chart' && (
            <DataChart 
              data={chartData} 
              pair={filters.pair}
              interval={filters.interval_type}
            />
          )}
        </div>
      </div>
    </div>
  );
}

export default App;