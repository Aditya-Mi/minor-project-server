# notification_manager.py
from firebase_admin import credentials, messaging, initialize_app
import logging

class NotificationManager:
    def __init__(self, service_account_path):
        try:
            # Initialize Firebase Admin SDK
            cred = credentials.Certificate(service_account_path)
            initialize_app(cred)
            
            # Configure logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            
            # Predefined fire alert message
            self.FIRE_ALERT = {
                'title': '🚨 FIRE ALERT!',
                'body': 'Fire detected! Please evacuate immediately following safety protocols.',
                'data': {
                    'type': 'fire_alert',
                    'priority': 'high',
                    'action': 'evacuate'
                }
            }
            
            self.logger.info("NotificationManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize NotificationManager: {str(e)}")
            raise

    def send_fire_alert(self, fcm_token):
        """
        Send fire alert notification to a specific device
        """
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=self.FIRE_ALERT['title'],
                    body=self.FIRE_ALERT['body']
                ),
                data=self.FIRE_ALERT['data'],
                token=fcm_token,
                android=messaging.AndroidConfig(
                    priority='high',
                    notification=messaging.AndroidNotification(
                        priority='max',
                        sound='emergency',
                        channel_id='fire_alert_channel'
                    )
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            sound='emergency.aiff',
                            category='FIRE_ALERT'
                        )
                    )
                )
            )

            response = messaging.send(message)
            self.logger.info(f"Successfully sent fire alert to token: {fcm_token[:10]}...")
            return True, response
            
        except messaging.ApiCallError as firebase_error:
            self.logger.error(f"Firebase API error: {str(firebase_error)}")
            return False, str(firebase_error)
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            return False, str(e)