IGNORED_WARNINGS = -Wno-unused-result -Wno-uninitialized -Wno-char-subscripts
CXX = g++
CXXFLAGS = -O2 -Wall -g -pg $(IGNORED_WARNINGS)
SOURCES_COMMON = main.cpp */*.cpp
EXCLUDE_DECIDE = code/decide.cpp
EXCLUDE_ANFF125_DECIDE = code/decide_anff125.cpp

agent_anff125: $(SOURCES_COMMON)
	$(CXX) $(filter-out $(EXCLUDE_DECIDE), $(SOURCES_COMMON)) decide_anff125.cpp -o $@ $(CXXFLAGS)

# Build the version using decide.cpp
agent_decide: $(SOURCES_COMMON)
	$(CXX) $(filter-out $(EXCLUDE_ANFF125_DECIDE), $(SOURCES_COMMON)) decide.cpp -o $@ $(CXXFLAGS)

# Default target
all: agent_anff125 agent_decide

# Clean build files
clean:
	rm -f agent_anff125 agent_decide
