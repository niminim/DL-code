import logging

logging.basicConfig(filename='app.log',
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


# Step 3: Logging Messages
# Use different methods of the logging module to log messages at different severity levels:
# debug, info, warning, error, and critical.
logging.debug("This is a debug message")
logging.info("Informational message")
logging.warning("Warning occurs")
logging.error("An error has happened!")
logging.critical("Critical issue")


# Step 4: Logger Instances
# For large applications, you might want to create separate logger instances for different parts of the application:
logger = logging.getLogger('my_logger')
logger.info("This is an info message from my_logger")


# Step 5: Logging in Modules
# If your application spans multiple modules, it's a good practice to create a dedicated logger in each module:
# In your module.py
logger = logging.getLogger(__name__)

def some_function():
    logger.info("Message from some_function")
# Using __name__ ensures that the logger name matches the module name.

# Step 7: Handling Exceptions
# To log exceptions, use logging.exception within an exception handler. It logs a message with level ERROR and adds exception information:
try:
    1 / 0
except ZeroDivisionError:
    logging.exception("Exception occurred")

