# bart-Xai-AND-t5-SUMMARY-
Explainable AI : Captum's - Integrated gradient  for Bart and normal Text summarization For t5


Integrated gradient to see token level attributions that infact is acting upon to make the summary and the value withrespect to the baseline of the general concept or the overview for Bart and normal Text summarization For t5[XAI for t5 is bit heavy to work on as the model consists encoder and decoder structure both which makes the model more efficient for the text generation and downstream tasks]

Issues faced :
for bart's XAI : bertviz wasnt working but captum did!, using the generate function worked more effectively than summarize
for T5's XAI : Model crashed constantly while tokenization and wasnt able to provide any output 
Future work : To apply XAI on T5 to understand the black box of the model
