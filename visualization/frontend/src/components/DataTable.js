import React from 'react';

const DataTable = ({ data, filters, onPageChange }) => {
  const formatNumber = (num) => {
    if (num === null || num === undefined) return 'N/A';
    return parseFloat(num).toFixed(5);
  };

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return dateString;
    }
  };

  const handlePrevPage = () => {
    if (filters.page > 1) {
      onPageChange(filters.page - 1);
    }
  };

  const handleNextPage = () => {
    if (data.length === filters.limit) {
      onPageChange(filters.page + 1);
    }
  };

  if (!data || data.length === 0) {
    return <div className="loading">No data available</div>;
  }

  return (
    <div>
      <h3>Data Table</h3>
      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Pair</th>
              <th>Interval</th>
              <th>Open</th>
              <th>High</th>
              <th>Low</th>
              <th>Close</th>
              <th>Volume</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, index) => (
              <tr key={index}>
                <td>{formatDate(row.timestamp)}</td>
                <td>{row.pair}</td>
                <td>{row.interval_type}</td>
                <td>{formatNumber(row.open)}</td>
                <td>{formatNumber(row.high)}</td>
                <td>{formatNumber(row.low)}</td>
                <td>{formatNumber(row.close)}</td>
                <td>{row.volume || 0}</td>
                <td>{row.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <button 
          onClick={handlePrevPage} 
          disabled={filters.page <= 1}
          className="button"
        >
          Previous
        </button>
        
        <span>Page {filters.page}</span>
        
        <button 
          onClick={handleNextPage} 
          disabled={data.length < filters.limit}
          className="button"
        >
          Next
        </button>
      </div>

      <div style={{ marginTop: '10px', fontSize: '14px', color: '#666' }}>
        Showing {data.length} records (Page {filters.page})
      </div>
    </div>
  );
};

export default DataTable;