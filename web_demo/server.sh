#!/bin/bash
# Server management script for web_demo

PID_FILE="server.pid"
LOG_FILE="app.log"
CONFIG_FILE="${2:-configs/stream.yaml}"  # Default config file

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
        
        echo "Starting server..."
        echo "Config file: $CONFIG_FILE"
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate motion_gen
        nohup python app.py --config "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
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
        $0 start "$CONFIG_FILE"
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
        echo "Usage: $0 {start|stop|restart|status} [config_file]"
        echo ""
        echo "Commands:"
        echo "  start [config_file]  - Start the server with optional config file"
        echo "  stop                 - Stop the server"
        echo "  restart [config_file]- Restart the server with optional config file"
        echo "  status               - Check server status"
        echo ""
        echo "Examples:"
        echo "  $0 start                           # Use default config (configs/stream.yaml)"
        echo "  $0 start configs/stream_tiny.yaml  # Use custom config"
        echo "  $0 restart configs/stream_tiny.yaml  # Restart with custom config"
        exit 1
        ;;
esac

