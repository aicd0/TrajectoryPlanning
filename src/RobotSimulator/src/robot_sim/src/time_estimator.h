#include <type_traits>
#include <limits>
#include <algorithm>

class TimeEstimator {
  unsigned int m_base;
  unsigned int m_elapsed = 0;
  unsigned int m_step = 1;
  bool m_upward = true;

public:
  TimeEstimator(const unsigned int& initial_val) :
    m_base(initial_val) {}

  void commit(unsigned int increment, const bool& completed) {
    if (m_elapsed > ~increment) {
      increment = ~m_elapsed;
    }
    m_elapsed += increment;

    if (completed) {
      m_base = std::min(m_base, m_elapsed);
    }
    else {
      m_base = std::max(m_base, m_elapsed);
    }

    if (m_elapsed == m_base || completed == (m_elapsed < m_base)) {
      if (m_upward != completed) {
        m_step += m_step > ~m_step ? ~m_step : m_step;
      }
      else {
        m_upward = !completed;
        m_step = 1;
      }

      if (m_upward) {
        m_base += m_base > ~m_step ? ~m_base : m_step;
      }
      else {
        m_base -= m_base < m_step ? m_base : m_step;
      }
    }

    if (completed) {
      m_elapsed = 0;
    }
  }

  inline unsigned int elapsed() const {
    return m_elapsed;
  }

  unsigned int total() const {
    return m_base;
  }

  inline unsigned int remain() const {
    return m_base > m_elapsed ? m_base - m_elapsed : 0;
  }
};
