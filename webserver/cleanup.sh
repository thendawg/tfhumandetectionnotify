#!/bin/bash

find /var/www/html/detectimgs -ctime +5 -exec rm -rf {} +
