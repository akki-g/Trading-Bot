import React from 'react';

const StatsSummary = ({ statistics }) => {
  if (!statistics) {
    return <div>No statistics available</div>;
  }

  const formatNumber = (num) => {
    if (num === null || num === undefined) return 'N/A';
    return parseFloat(num).toFixed(5);
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      return new Date(dateString).toLocaleDateString();
    } catch {
      return dateString;
    }
  };

  const formatInteger = (num) => {
    if (num === null || num === undefined) return 'N/A';
    return num.toLocaleString();
  };

  return (
    <div>
      <h3>Data Statistics</h3>
      <div className="stats-grid">
        <div className="stat-item">
          <div className="stat-value">{formatInteger(statistics.record_count)}</div>
          <div className="stat-label">Total Records</div>
        </div>
        
        <div className="stat-item">
          <div className="stat-value">{formatDate(statistics.earliest_date)}</div>
          <div className="stat-label">Earliest Date</div>
        </div>
        
        <div className="stat-item">
          <div className="stat-value">{formatDate(statistics.latest_date)}</div>
          <div className="stat-label">Latest Date</div>
        </div>
        
        <div className="stat-item">
          <div className="stat-value">{formatNumber(statistics.avg_price)}</div>
          <div className="stat-label">Average Price</div>
        </div>
        
        <div className="stat-item">
          <div className="stat-value">{formatNumber(statistics.min_price)}</div>
          <div className="stat-label">Minimum Price</div>
        </div>
        
        <div className="stat-item">
          <div className="stat-value">{formatNumber(statistics.max_price)}</div>
          <div className="stat-label">Maximum Price</div>
        </div>
        
        <div className="stat-item">
          <div className="stat-value">{formatNumber(statistics.price_stddev)}</div>
          <div className="stat-label">Price Std Dev</div>
        </div>
      </div>
    </div>
  );
};

export default StatsSummary;