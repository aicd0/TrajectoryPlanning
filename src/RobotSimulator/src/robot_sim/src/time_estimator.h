#include <type_traits>
#include <limits>
#include <algorithm>

class TimeEstimator {
  unsigned int m_total;
  unsigned int m_elapsed = 0;
  unsigned int m_step = 1;
  bool m_upward = true;

public:
  TimeEstimator(const unsigned int& initial_val) :
    m_total(initial_val) {}

  void commit(unsigned int increment, const bool& completed) {
    if (m_elapsed > ~increment) {
      increment = ~m_elapsed;
    }
    m_elapsed += increment;

    if (completed) {
      m_total = std::min(m_total, m_elapsed);
    }
    else {
      m_total = std::max(m_total, m_elapsed);
    }

    if (m_elapsed == m_total || completed == (m_elapsed < m_total)) {
      if (m_upward != completed) {
        m_step += m_step > ~m_step ? ~m_step : m_step;
      }
      else {
        m_upward = !completed;
        m_step = 1;
      }

      if (m_upward) {
        m_total += m_total > ~m_step ? ~m_total : m_step;
      }
      else {
        m_total -= m_total < m_step ? m_total : m_step;
      }
    }

    if (completed) {
      m_elapsed = 0;
    }
  }

  inline unsigned int elapsed() const {
    return m_elapsed;
  }

  inline unsigned int total() const {
    return m_total;
  }

  inline unsigned int remain() const {
    return m_total > m_elapsed ? m_total - m_elapsed : 0;
  }
};
