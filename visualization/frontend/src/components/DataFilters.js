import React from 'react';

const DataFilters = ({ filters, availablePairs, availableIntervals, onFilterChange }) => {
  const handleInputChange = (field, value) => {
    onFilterChange({ [field]: value });
  };

  return (
    <div className="filters">
      <div className="filter-group">
        <label htmlFor="pair">Currency Pair</label>
        <select
          id="pair"
          value={filters.pair}
          onChange={(e) => handleInputChange('pair', e.target.value)}
        >
          <option value="">Select Pair</option>
          {availablePairs.map(pair => (
            <option key={pair} value={pair}>{pair}</option>
          ))}
        </select>
      </div>

      <div className="filter-group">
        <label htmlFor="interval">Interval</label>
        <select
          id="interval"
          value={filters.interval_type}
          onChange={(e) => handleInputChange('interval_type', e.target.value)}
        >
          <option value="">Select Interval</option>
          {availableIntervals.map(interval => {
            // Handle both old format (string) and new format (object)
            const value = typeof interval === 'string' ? interval : interval.value;
            const label = typeof interval === 'string' ? interval : interval.label;
            return (
              <option key={value} value={value}>{label}</option>
            );
          })}
        </select>
      </div>

      <div className="filter-group">
        <label htmlFor="start_date">Start Date</label>
        <input
          type="date"
          id="start_date"
          value={filters.start_date}
          onChange={(e) => handleInputChange('start_date', e.target.value)}
        />
      </div>

      <div className="filter-group">
        <label htmlFor="end_date">End Date</label>
        <input
          type="date"
          id="end_date"
          value={filters.end_date}
          onChange={(e) => handleInputChange('end_date', e.target.value)}
        />
      </div>

      <div className="filter-group">
        <label htmlFor="limit">Records Limit</label>
        <select
          id="limit"
          value={filters.limit}
          onChange={(e) => handleInputChange('limit', parseInt(e.target.value))}
        >
          <option value={100}>100</option>
          <option value={500}>500</option>
          <option value={1000}>1,000</option>
          <option value={5000}>5,000</option>
          <option value={10000}>10,000</option>
          <option value={50000}>50,000</option>
          <option value={100000}>100,000</option>
        </select>
      </div>

      <div className="filter-group">
        <label>Quick Date Ranges</label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
          <button
            className="button"
            style={{ fontSize: '12px', padding: '5px 8px' }}
            onClick={() => {
              const today = new Date();
              const lastMonth = new Date(today.getFullYear(), today.getMonth() - 1, today.getDate());
              onFilterChange({
                start_date: lastMonth.toISOString().split('T')[0],
                end_date: today.toISOString().split('T')[0],
                page: 1
              });
            }}
          >
            Last Month
          </button>
          <button
            className="button"
            style={{ fontSize: '12px', padding: '5px 8px' }}
            onClick={() => {
              const today = new Date();
              const lastYear = new Date(today.getFullYear() - 1, today.getMonth(), today.getDate());
              onFilterChange({
                start_date: lastYear.toISOString().split('T')[0],
                end_date: today.toISOString().split('T')[0],
                page: 1
              });
            }}
          >
            Last Year
          </button>
          <button
            className="button"
            style={{ fontSize: '12px', padding: '5px 8px' }}
            onClick={() => {
              onFilterChange({
                start_date: '2020-06-29',
                end_date: '2025-06-28',
                page: 1
              });
            }}
          >
            All Data (2020-2025)
          </button>
          <button
            className="button"
            style={{ fontSize: '12px', padding: '5px 8px' }}
            onClick={() => {
              onFilterChange({
                start_date: '',
                end_date: '',
                limit: 1000,
                page: 1
              });
            }}
          >
            Reset Filters
          </button>
        </div>
      </div>
    </div>
  );
};

export default DataFilters;