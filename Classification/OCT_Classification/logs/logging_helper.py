import os.path
import logging

def get_logger(config):
# Basic logging setup
    logging.basicConfig(filename= os.path.join(config['logging']['logfile_path'],'training.log'),
                        filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)
    return logger

def close_logger(logger):
    for handler in logger.handlers[:]:  # Iterate over a copy of the handler list
        handler.close()  # Close the handler
        logger.removeHandler(handler)  # Remove the handler from the logger
    logging.shutdown()  # Optional: Shuts down the logging system entirely


#####
def get_logger_update_saved_file(config):

    from logging.handlers import RotatingFileHandler

    class ImmediateFileHandler(RotatingFileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    # Define a custom logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler and set it to use the custom handler
    file_handler = ImmediateFileHandler(os.path.join(config['logging']['logfile_path'],'training.log'),
                                        mode='w', maxBytes=5*1024*1024,
                                        backupCount=2)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def close_log_file(logger, handler):
    # Flush and close the handler
    handler.close()
    # Remove the handler from the logger
    logger.removeHandler(handler)

# When you need to close the file
# close_log_file(logger, file_handler)