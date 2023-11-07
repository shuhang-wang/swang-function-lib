import wandb
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def read_google_sheet(filename, sheetname):
    '''
    Readd google doc sheet
    '''
    # Use artifact for accessing Google Sheets credentials
    api = wandb.Api()
    artifact = api.artifact('cellino-ml-ninjas/constants/cred_cell_id_google_sheet:latest', type='credentials')
    artifact_dir = artifact.download()
    cred_path = pj(artifact_dir, 'cred_cell_id_google_sheet.json')

    # Define the scope for Google Sheets API
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    # Authenticate with the service account
    creds = ServiceAccountCredentials.from_json_keyfile_name(cred_path, scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet and select the right worksheet
    spreadsheet = client.open(filename)
    sheet = spreadsheet.worksheet(sheetname)

    return sheet