"use client";
import { useState, useRef } from "react";
import {
  Box, Grid, VStack, HStack, Text, Button, Textarea,
  Badge, Heading, Separator, Wrap, WrapItem, Card
} from "@chakra-ui/react";

export default function Home() {
  const [recording, setRecording] = useState(false);
  const [score, setScore] = useState<number | null>(null);
  const [mistakes, setMistakes] = useState<string[]>([]);
  const [tip, setTip] = useState("");
  const [transcript, setTranscript] = useState("");

  const targetText = "Ich möchte bitte eine Flasche Mineralwasser.";
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const handleRecord = async () => {
    if (!recording) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.start();
      setRecording(true);
    } else {
      mediaRecorderRef.current?.stop();
      mediaRecorderRef.current!.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", audioBlob, "recording.webm");
        formData.append("target_text", targetText); // ✅ khớp backend

        try {
          const res = await fetch("http://localhost:8000/api/evaluate", {
            method: "POST",
            body: formData,
          });
          const data = await res.json();

          setScore(data.score ?? null);
          setMistakes(Array.isArray(data.mistakes) ? data.mistakes : []);
          setTranscript(data.transcript ?? "");
          setTip(data.tip ?? "");
        } catch (err) {
          console.error("API error", err);
        }
      };
      setRecording(false);
    }
  };

  return (
    <Box minH="100vh" p={8} display="flex" alignItems="center" justifyContent="center">
      <Grid templateColumns={{ base: "1fr", md: "repeat(2, 1fr)" }} gap={8} maxW="6xl" w="100%">

        {/* Left */}
        <Card.Root p={6} rounded="2xl" shadow="sm" borderWidth="1px">
          <Card.Header pb={2}>
            <Heading size="md">🎤 Pronunciation Practice</Heading>
          </Card.Header>
          <Card.Body>
            <VStack align="start" gap={5} w="full">
              <Text fontSize="xl" fontWeight="semibold">{targetText}</Text>
              <HStack gap={4}>
                <Button
                  onClick={handleRecord}
                  colorScheme={recording ? "red" : "gray"}
                  rounded="full"
                  w="56px" h="56px"
                >🎙</Button>
                <Text fontSize="sm">{recording ? "Recording…" : "Tap to record"}</Text>
              </HStack>
              <Textarea value={transcript} readOnly rows={3} />
            </VStack>
          </Card.Body>
        </Card.Root>

        {/* Right */}
        <Card.Root p={6} rounded="2xl" shadow="sm" borderWidth="1px">
          <Card.Header pb={2}>
            <Heading size="md">📊 Score & Feedback</Heading>
          </Card.Header>
          <Card.Body>
            {score !== null ? (
              <VStack align="start" gap={5} w="full">
                <Box>
                  <Text fontSize="4xl" fontWeight="bold">{score}%</Text>
                </Box>
                <VStack align="start" gap={2} w="full">
                  <Text fontWeight="medium">Mistake words</Text>
                  <Wrap gap={2}>
                    {mistakes.length > 0 ? (
                      mistakes.map((w, i) => (
                        <WrapItem key={i}><Badge colorScheme="red">{w}</Badge></WrapItem>
                      ))
                    ) : <Text color="gray.400">No mistakes 🎉</Text>}
                  </Wrap>
                </VStack>
                <Separator />
                <Text fontWeight="medium">Tip</Text>
                <Text>{tip}</Text>
              </VStack>
            ) : (
              <Text color="gray.400">No score yet. Record and analyze.</Text>
            )}
          </Card.Body>
        </Card.Root>
      </Grid>
    </Box>
  );
}