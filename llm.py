import os

import json
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY") or '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',
)


def get_relevant_queries(ai_description, transcript, ocr, user_description):
    chat_completion = client.chat.completions.create(
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": 'Respond with a JSON in the following format: {"queries": ["query one", "query two", ...]}. Queries must be in Russian language. Queries must be short, concise and simple, yet informative.'
            },
            {
                "role": "user",
                "content": f"""
    Напиши 5-10 поисковых запросов НА РУССКОМ ЯЗЫКЕ, которые могут быть связаны с видео.
    Вот описание того, что происходит в видео: "{ai_description}"
    Вот транскрипт всех слов, которые произносятся в видео: "{transcript}"
    """,
            }
        ],    
        
        model="llama3-8b-8192",
    )
    result = chat_completion.choices[0].message.content
    print(result)


    return json.loads(result)['queries']

# get_relevant_queries("in the game a character rides a bicycle and then jumps off the bike",
#                      """
# Сейчас я тебе покажу секретную команду в Roblox. Чтобы её активировать, поставь лайк, подпишись, а т
#     кже в виде в чат команду I love you. Когда вы её ведете, у вас на экране появится вот такой clone. 
#     крыт был создан одним из создателей Roblox и работает только в игре, где есть Roblox Diva Loper Service. Чтобы вы раскрымиран же всего-то выйти из игры, но лучше не проверяйся.
#     """
#                      )