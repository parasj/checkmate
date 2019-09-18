#!/bin/sh
source .env
redis-cli -u redis://$REDIS_PASSWORD@$REDIS_HOST:$REDIS_PORT/$REDIS_DB