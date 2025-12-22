#!/bin/bash
# Server management script for web_demo

PID_FILE="server.pid"
LOG_FILE="app.log"
GPU_ID="${2:-0}"  # Default GPU ID is 0, can be overridden by second argument

case "$1" in
    start)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "Server is already running (PID: $PID)"
                exit 1
            else
                echo "Removing stale PID file"
                rm -f "$PID_FILE"
            fi
        fi
        
        echo "Using GPU: $GPU_ID"
        echo "Starting server..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate motion_gen
        CUDA_VISIBLE_DEVICES=$GPU_ID nohup python app.py > "$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 3
        
        if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
            echo "Server started successfully (PID: $(cat $PID_FILE))"
            curl -s http://localhost:5000/api/status
        else
            echo "Failed to start server"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;
        
    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "No PID file found. Killing all python app.py processes..."
            pkill -9 -f "python app.py"
            exit 0
        fi
        
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "Stopping server (PID: $PID)..."
            kill -9 $PID
            rm -f "$PID_FILE"
            echo "Server stopped"
        else
            echo "Server is not running"
            rm -f "$PID_FILE"
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start $GPU_ID
        ;;
        
    status)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if ps -p $PID > /dev/null 2>&1; then
                echo "Server is running (PID: $PID)"
                curl -s http://localhost:5000/api/status
            else
                echo "PID file exists but process is not running"
                rm -f "$PID_FILE"
                exit 1
            fi
        else
            echo "Server is not running"
            exit 1
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|restart|status} [GPU_ID]"
        echo "  GPU_ID: optional, default is 0"
        echo "  Example: $0 start 3  # Use GPU 3"
        exit 1
        ;;
esac

